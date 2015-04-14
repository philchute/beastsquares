import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

object Main{
   val conf = new SparkConf().setAppName("als")
   // Create a Scala Spark Context.
   val sc = new SparkContext(conf)
   val rawUserArtistData = sc.textFile("s3n://beastsquares/user_artist_data.txt")
   rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
   rawUserArtistData.map(_.split(' ')(1).toDouble).stats()

   val rawArtistData = sc.textFile("s3n://beastsquares/artist_data.txt")
   val artistByID = rawArtistData.flatMap { line =>
       val (id, name) = line.span(_ != '\t')
       if (name.isEmpty) {
           None
       } else {
           try {
               Some((id.toInt, name.trim))
           } catch {
               case e: NumberFormatException => None
            }
       }
   }

   val rawArtistAlias = sc.textFile("s3n://beastsquares/artist_alias.txt")
   val artistAlias = rawArtistAlias.flatMap { line =>
       val tokens = line.split('\t')
       if (tokens(0).isEmpty) {
           None
       } else {
           Some((tokens(0).toInt, tokens(1).toInt))
       }
   }.collectAsMap()

   //foo.head gets the first element.
   artistAlias.head
   artistByID.lookup(6803336).head
   artistByID.lookup(1000010).head

   import org.apache.spark.mllib.recommendation._
   val bArtistAlias = sc.broadcast(artistAlias)
   val trainData = rawUserArtistData.map { line =>
       val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
       val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
       Rating(userID, finalArtistID, count)
   }.cache( )

   val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
   model.userFeatures.mapValues(_.mkString(", ")).first()

   //1000019

   val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).
            filter { case Array(user,_,_) => user.toInt == 1000019 }

   val existingProducts =
      rawArtistsForUser.map { case Array(_,artist,_) => artist.toInt }.
      collect().toSet

   artistByID.filter { case (id, name) =>
       existingProducts.contains(id)
   }.values.collect().foreach(println)

   val recommendations = model.recommendProducts(1000019, 5)
   recommendations.foreach(println)

   val recommendedProductIDs = recommendations.map(_.product).toSet
   artistByID.filter { case (id, name) =>
          recommendedProductIDs.contains(id)
   }.values.collect().foreach(println)

   //AUC
   def areaUnderCurve(
      positiveData: RDD[Rating],
      bAllItemIDs: Broadcast[Array[Int]],
      predictFunction: (RDD[(Int,Int)] => RDD[Rating])) = {

      val positiveUserProducts = positiveData.map(r => (r.user, r.product))

      val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

      val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {

      userIDAndPosItemIDs => {
        val random = new Random()
        val allItemIDs = bAllItemIDs.value
        userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0
          while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.size))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }

          negative.map(itemID => (userID, itemID))
        }
      }
      }.flatMap(t => t)

      val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

      positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>

        var correct = 0L
        var total = 0L

        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }

        correct.toDouble / total
      }.mean() // Return mean AUC over users
   }

   def buildRatings(
         rawUserArtistData: RDD[String],
         bArtistAlias: Broadcast[Map[Int,Int]]) = {
       rawUserArtistData.map { line =>
         val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
         val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
         Rating(userID, finalArtistID, count)
       }
   }

   val allData = buildRatings(rawUserArtistData, bArtistAlias)
   val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
   trainData.cache()
   cvData.cache()

   val allItemIDs = allData.map(_.product).distinct().collect()
   val bAllItemIDs = sc.broadcast(allItemIDs)
   val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
   val auc1 = areaUnderCurve(cvData, bAllItemIDs, model.predict)

   def predictMostListened(sc: SparkContext, train: RDD[Rating])(allData: RDD[(Int,Int)]) = {
       val bListenCount =
         sc.broadcast(train.map(r => (r.product, r.rating)).reduceByKey(_ + _).collectAsMap())
       allData.map { case (user, product) =>
         Rating(user, product, bListenCount.value.getOrElse(product, 0.0))
       }
     }

   val auc2 = areaUnderCurve(cvData, bAllItemIDs, predictMostListened(sc, trainData))


   //val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
   val someUsers = allData.map(_.user).distinct().take(20)

   val someRecommendations = someUsers.map(userID => model.recommendProducts(userID, 5))
   someRecommendations.map(
       recs => recs.head.user + " -> " + recs.map(_.product).mkString(", ")
   ).foreach(println)

   //parameter search

   def unpersist(model: MatrixFactorizationModel): Unit = {
       model.userFeatures.unpersist()
       model.productFeatures.unpersist()
     }

   //may throw ava.lang.OutOfMemoryError: Java heap space if RAM is not enough
   val evaluations = for (rank   <- Array(10,  50);
                          lambda <- Array(1.0, 0.0001);
                          alpha  <- Array(1.0, 40.0))
      yield {
         val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
         val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
         unpersist(model)
         println( rank + " " + lambda + " " + alpha + " " + auc)
         ((rank, lambda, alpha), auc)
      }
   evaluations.sortBy(_._2).reverse.foreach(println)
}
