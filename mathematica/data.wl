(* ::Package:: *)

BeginPackage["mimir`data`"]

makePoly::usage = ""
addBiasTerm::usage = ""
binarize::usage = ""
stratifiedSample::usage = ""
mlclassEx4::usage = ""
mlclassEx5::usage = ""
iris::usage = ""

Begin["`Private`"]

makePoly[x_, n_:6] :=
  Flatten@Table[#[[1]]^i #[[2]]^(j-i),{i,0,n},{j,i,n}]& /@ x

addBiasTerm[x_] :=
  Prepend[#, 1.] & /@ x

factorToIndicator[x_, factor_] :=
  If[Equal[factor, #], 1., 0.] & /@ x

binarize[x_] :=
  Module[{factors = DeleteDuplicates[x]},
    (factorToIndicator[x, #] & /@ factors)\[Transpose]]

stratifiedSample[y_, ratio_:.8] :=
  MapThread[Join,
    Module[{idx = Flatten@Position[#, 1.0]},
     Module[{sample = RandomSample[idx, Ceiling[ratio*Length@idx]]},
      {sample, Complement[idx, sample]}]] & /@ (y\[Transpose])]

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

End[]
EndPackage[]



