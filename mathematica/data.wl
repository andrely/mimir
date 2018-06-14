(* ::Package:: *)

BeginPackage["mimir`data`"];

makePoly::usage = "";
addBiasTerm::usage = "";
binarize::usage = "";
normalize::usage = "";
stratifiedSample::usage = "";

checkOneHotArray::usage = "Checks if array is a valid one-hot array.";

mlclassEx4::usage = "";
mlclassEx5::usage = "";
iris::usage = "";
newsgroups::usage = "";

Begin["`Private`"];

makePoly[x_, n_:6] :=
  Flatten@Table[#[[1]]^i #[[2]]^(j-i),{i,0,n},{j,i,n}]& /@ x

addBiasTerm[x_] :=
  Prepend[#, 1.] & /@ x

factorToIndicator[x_, factor_] :=
  If[Equal[factor, #], 1., 0.] & /@ x

binarize[x_] :=
  Module[{factors = DeleteDuplicates[x]},
    (factorToIndicator[x, #] & /@ factors)\[Transpose]]

normalize[x_] :=
  Module[{mu = Mean[x], s = Variance[x]},
    (# - mu) / s & /@ x]

stratifiedSample[y_, ratio_:.8] :=
  MapThread[Join,
    Module[{idx = Flatten@Position[#, 1.0]},
     Module[{sample = RandomSample[idx, Ceiling[ratio*Length@idx]]},
      {sample, Complement[idx, sample]}]] & /@ (y\[Transpose])]

(* Alternative taking list of class names *)
stratifiedSplit[groups_, ratio_:.8] :=
  Module[{c = DeleteDuplicates[groups], subSplits},
    subSplits = Map[
      Module[{pos = RandomSample@Flatten@Position[groups, #], idx},
        idx = Ceiling[Length@pos*ratio];
        {pos[[;;idx]], pos[[idx;;]]}] &,
      c];
    Apply[Join,#]&/@(subSplits\[Transpose])]
    
checkOneHotArray[a_]:=
  ArrayDepth@a == 2 && Total[a, 2] == Length@a && Sort@DeleteDuplicates@Flatten@a == {0, 1}

mlclassEx4[path_]:=
  {ex4Data[path,"x"],ex4Data[path,"y"]}

ex4Data[path_,"x"]:=
  Module[{x=Import[StringJoin[path, $PathnameSeparator,"ex4x.dat"]]},
    MapThread[Join,{ConstantArray[1,{Dimensions[x,1][[1]],1}],x}]]

ex4Data[path_,"y"]:=
  Module[{y=Import[StringJoin[path, $PathnameSeparator,"ex4y.dat"]]},
    MapThread[Join,{y,1-y}]]

mlclassEx5[path_, "x"] :=
  Import[StringJoin[path, $PathnameSeparator,"ex5Logx.dat"], "CSV"]

mlclassEx5[path_, "y"] :=
  Flatten@Import[StringJoin[path, $PathnameSeparator,"ex5Logy.dat"], "CSV"]

iris["x"] :=
  Part[#, 1] & /@ ExampleData[{"MachineLearning","FisherIris"},"Data"]

iris["y"] :=
  Part[#, 2]\[NonBreakingSpace]& /@ ExampleData[{"MachineLearning","FisherIris"},"Data"]

removeHeader[art_]:=StringRiffle[StringExtract[art,"\n\n"->2;;All],"\n\n"]

removeQuotes[art_] := 
  StringRiffle[Select[StringExtract[removeHeader[art], "\n" -> All],
    Not[StringContainsQ[#, RegularExpression["(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)"]]] &], "\n"]

removeSignature[art_] :=
  Module[{lines=Reverse@StringSplit[art,"\n"]},
    lines=Drop[lines,FirstPosition[lines,Except[""]][[1]]];
    Module[{otherLines=StringTrim@StringReplace[#,"-"->""]&/@lines},
      StringRiffle[Reverse[Drop[lines,FirstPosition[otherLines,""][[1]]]],"\n"]]]

newsgroups[path_] :=
  Module[{groups = Last@FileNameSplit[#] & /@ FileNames@ToFileName[path, "*"]},
    Module[{data=Flatten[Distribute[{#, removeSignature@removeQuotes@removeHeader@Import[#,"Text"]&/@FileNames@ToFileName[{path,groups[[1]]},"*"]},List]&/@groups,1]\[Transpose]},
      {"x"->data[[2]], "y"->data[[1]]}]]

End[];
EndPackage[];



