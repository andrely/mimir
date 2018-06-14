name := "scala"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += "Artima Maven Repository" at "http://repo.artima.com/releases"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",

  "org.scala-lang" % "scala-reflect" % "2.11.8",
  "org.scala-lang.modules" %% "scala-xml" % "1.0.4",
  "org.scalactic" %% "scalactic" % "2.2.6",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test"
)
