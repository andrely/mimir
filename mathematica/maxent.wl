(* ::Package:: *)

BeginPackage["mimir`maxent`"];

logprob::usage = "";
prob::usage = "";
cost::usage = "";
grad::usage = "";
hessian::usage = "";
update::usage = "";
train::usage = "";
predict::usage = "";

Begin["`Private`"];

logSumExp[x_] :=
  Module[{m = Max[x]},
    m + Log[Total[Exp[x - m]]]]

act[x_, \[Theta]_] := 
  MapThread[Join, {Exp[x.\[Theta]\[Transpose]], ConstantArray[1,{Dimensions[x,1][[1]],1}]}] // N

prob[x_, \[Theta]_] :=
  Module[{a = act[x, \[Theta]]},
    Module[{z = Total /@ a},
      MapThread[Divide,{a, z}] // N]]

logprob[x_, \[Theta]_] :=
  # - logSumExp[#] & /@ (Append[(x.\[Theta]\[Transpose])\[Transpose], ConstantArray[0, Dimensions[x, 1]]]\[Transpose])

cost[x_, y_, \[Theta]_, \[Lambda]_:1.0] :=
  Module[{n = Dimensions[x][[1]]},
    (Total[-y*logprob[x,\[Theta]], 2] / n) + \[Lambda]*Total[\[Theta][[All,2;;]]^2,2] / (2*n)]

grad[x_, y_, \[Theta]_, \[Lambda]_:1.0] :=
  Module[{err = Exp[logprob[x,\[Theta]]] - y, n = Dimensions[x][[1]], c = Dimensions[y][[2]]},
    Module[{reg = (\[Lambda]/Dimensions[x][[1]]) * Join[ConstantArray[0., {1, c-1}], \[Theta][[All, 2;;]]\[Transpose]]\[Transpose]},
      Table[Total@MapThread[#1 * #2 + reg[[i, All]] &, {err[[All,i]], x}] / n // N, {i, 1, c-1}]]]

hessian[x_, \[Theta]_, \[Lambda]_:1.0] :=
  Module[{n = Dimensions[x][[1]], c = Dimensions[\[Theta]][[1]]},
    Module[{\[Mu] = Exp[logprob[x,\[Theta]]][[All,1;;c]], 
            reg = (\[Lambda] / n) * DiagonalMatrix[
                    Flatten@Table[Prepend[ConstantArray[1.,Dimensions[\[Theta]][[2]]-1],0.],Dimensions[\[Theta]][[1]]]]},
      (Total@MapThread[
        KroneckerProduct[
          DiagonalMatrix[#1] - Outer[Times,#1,#1],
          Outer[Times, #2, #2]] &,
        {\[Mu],x}] / n) + reg]]

update[x_, y_, \[Theta]_, \[Rho]_:.05, \[Lambda]_:1.0] :=
  Module[{h = hessian[x, \[Theta], \[Lambda]], g = grad[x, y, \[Theta], \[Lambda]]},
    \[Theta] - \[Rho]*ArrayReshape[Inverse[h].{Flatten[g]}\[Transpose], Dimensions[\[Theta]]]]

trainInternal[x_,y_,\[Lambda]_:1.0,iterations_:50]:=
  Module[{\[Theta] = RandomVariate[NormalDistribution[0,.001], {Dimensions[y][[2]]-1,Dimensions[x][[2]]}],
          stats={}},    
    Module[{j = 0, newJ = cost[x,y,\[Theta],\[Lambda]],iter=0},
      While[Abs[j - newJ] > .00001 && iter < iterations,
        iter++;
        Print[{iter,newJ}];
        AppendTo[stats, {iter,newJ}];
        j = newJ;
        \[Theta] = update[x,y,\[Theta],1.0,\[Lambda]];
        newJ = cost[x,y,\[Theta],\[Lambda]]];
      {"\[Theta]"->\[Theta], "stats"->stats}]]

train[x_,y_,\[Lambda]_:1.0,iterations_:50] :=
  Module[{result = trainInternal[x, y, \[Lambda], iterations]},
    {"type" -> "maxent", "\[Lambda]" -> \[Lambda], "iterations" -> iterations, "\[Theta]" -> ("\[Theta]" /. result), 
     "stats" -> ("stats" /. result)}]

predict[model_, x_] :=
  Flatten[Position[#,Max[#]] & /@ logprob[x, "\[Theta]" /. model]]

End[];
EndPackage[];
