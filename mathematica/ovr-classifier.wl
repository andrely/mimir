(* ::Package:: *)

BeginPackage["mimir`ovr`"]

ovrClassifier::usage = ""
train::usage = ""
predict::usage = ""

Begin["`Private`"]

ovrClassifier[trainFunc_, logProbFunc_] :=
  {"trainFunc" -> trainFunc, "logProbFunc" -> logProbFunc}

train[model_, x_, y_] :=
  Join[model, {"models" -> (("trainFunc" /. model)[x, #] & /@ (y\[Transpose]))}]

logProb[model_, x_] :=
  (("logProbFunc" /. model)[x, "\[Theta]" /. #] & /@ ("models"/. model))\[Transpose]

predict[model_, x_] :=
  Flatten[Ordering[#, -1] & /@ logProb[model, x]]

End[]
EndPackage[]



