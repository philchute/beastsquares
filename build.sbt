//import AssemblyKeys._

//assemblySettings

//jarName in assembly := "als.jar"

name := "ALS"

version := "0.1"

scalaVersion := "2.11.5"

//libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.0"
libraryDependencies ++= Seq(
  "org.apache.spark"  % "spark-core_2.10"              % "1.3.0" % "provided",
  "org.apache.spark"  % "spark-mllib_2.10"             % "1.3.0"
)
