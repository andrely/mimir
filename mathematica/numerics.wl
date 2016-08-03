(* ::Package:: *)

BeginPackage["mimir`numerics`"];

logSumExp::usage = "";

Begin["`Private`"];

logSumExp[m_] :=
  Module[{max = Max /@ m},
    max + Log[Total /@ Exp@MapThread[Subtract, {m, max}]]]

End[];
EndPackage[];
