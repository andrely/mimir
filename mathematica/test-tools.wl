(* ::Package:: *)

BeginPackage["mimir`testtools`"]

arrApproxEqual::usage = ""
approxEqual::usage = ""

Begin["`Private`"]

arrApproxEqual[x_,y_]:=(Total[Abs[x-y],2]/Length[x])<.001

approxEqual[x_,y_]:=Abs[x-y]<.001

End[]
EndPackage[]



