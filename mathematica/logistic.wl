(* ::Package:: *)

BeginPackage["mimir`logistic`"];

Begin["`Private`"];

a[x_, \[Theta]_] := Flatten[x.{\[Theta]}\[Transpose]]

logprob[x_, \[Theta]_] :=
  Module[{act = a[x, \[Theta]]},
    MapThread[Subtract, {act, mimir`numerics`logSumExp[Join[{act}, ConstantArray[0., {1, Length@act}]]\[Transpose]]}]]

prob[x_, \[Theta]_] := Exp[logprob[x, \[Theta]]]

cost[x_, y_, \[Theta]_ , \[Lambda]_:1.0]:=
  Module[{n = Dimensions[x][[1]]},
    (Total[-y*logprob[x, \[Theta]] - (1 - y)*(logprob[x, \[Theta]] - a[x, \[Theta]])]/n) + \[Lambda]*Total[\[Theta][[2;;]]^2]/(2*n)]

grad[x_, y_, \[Theta]_, \[Lambda]_:1.0] :=
  Module[{reg = (\[Lambda]/Dimensions[x][[1]]) * Join[{0.}, \[Theta][[2;;]]]},
	Total[MapThread[#1 * #2 + reg &, {Exp[logprob[x, \[Theta]]] - y, x}]] / Dimensions[x,1][[1]]]

hessian[x_,\[Theta]_,\[Lambda]_:1.0]:=
  Module[{n=Dimensions[x][[1]],
          p=Dimensions[x][[2]]},
    Module[{reg=DiagonalMatrix[Join[{0.},ConstantArray[\[Lambda]/n,p-1]]]},
      (Total@MapThread[Times,{Exp[2*logprob[x, \[Theta]] - a[x, \[Theta]]],Map[Outer[Times,#,#]&,x]}]/n) + reg]]

updateNewton[x_, y_, \[Theta]_, \[Lambda]_:1.0] :=
  Flatten[\[Theta] - Inverse[hessian[x, \[Theta], \[Lambda]]].{grad[x, y, \[Theta], \[Lambda]]}\[Transpose]]

updateSteep[x_, y_, \[Theta]_, \[Rho]_:.05, \[Lambda]_:1.0] :=
  \[Theta] - \[Rho] * grad[x, y, \[Theta], \[Lambda]]

updateSGD[x_, y_, \[Theta]_, \[Rho]_:.05, \[Lambda]_:1.0, batchSize_:10] :=
  Module[{sampleIdx = RandomSample[Range@Length@x], batchNum = Ceiling[Length@x / batchSize]},
    Module[{batchIdx = Table[{1 + batchSize * i, Min[batchSize + batchSize * i, Length@x]}, {i, 0, batchNum - 1}],
            xSample = x[[sampleIdx, All]], ySample = y[[sampleIdx]]},
      Fold[updateSteep[xSample[[#2, All]], ySample[[#2]], #1, \[Rho], \[Lambda]] &, \[Theta], batchIdx]]]

trainInternal[x_, y_, \[Lambda]_:1.0, \[Rho]_:.05, maxIter_:50]:=
  Module[{\[Theta] = RandomVariate[NormalDistribution[0,.001], Dimensions[x][[2]]],stats={}},    
    Module[{j = 0, newJ = cost[x,y,\[Theta],\[Lambda]],iter=0},
      While[Abs[j - newJ] > .00001 && iter < maxIter,
        iter++;
        (* Print[{iter,newJ}]; *)
        AppendTo[stats, {iter,newJ}];
        j = newJ;
        (* \[Theta] = updateSteep[x, y, \[Theta], \[Rho], \[Lambda]]; *)
        (* \[Theta] = updateSGD[x, y, \[Theta], \[Rho], \[Lambda], 20]; *)
        \[Theta] = updateNewton[x, y, \[Theta], \[Lambda]];
        newJ = cost[x,y,\[Theta],\[Lambda]]];
      {"\[Theta]"->\[Theta], "stats"->stats}]]

train[x_, y_, \[Lambda]_:1.0, \[Rho]_:.05, maxIter_:50] :=
  Module[{result = trainInternal[x, y, \[Lambda], \[Rho], maxIter]},
    {"type" -> "logistic", "\[Lambda]" -> \[Lambda], "\[Rho]" -> \[Rho], "maxIter" -> maxIter, "\[Theta]" -> ("\[Theta]" /. result), 
     "stats" -> ("stats" /. result)}]

predict[model_, x_] :=
  If[# > Log[.5], 1., 0.]& /@ logprob[x, "\[Theta]" /. model]

End[];
EndPackage[];
