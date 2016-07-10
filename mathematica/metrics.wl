(* ::Package:: *)

BeginPackage["mimir`metrics`"]

accuracy::usage = ""

Begin["`Private`"]

accuracy[true_, predicted_] :=
  Total@MapThread[If[Equal[#1, #2], 1., 0.] &, {true, predicted}] / Dimensions[true][[1]]

End[]
EndPackage[]



