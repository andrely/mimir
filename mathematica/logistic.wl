(* ::Package:: *)

BeginPackage["mimir`logistic`"]

prob::usage = ""
logprob::usage = ""
cost::usage = ""
grad::usage = ""
hessian::usage = ""
update::usage = ""
train::usage = ""
predict::usage = ""

Begin["`Private`"]

a[x_, \[Theta]_] := Flatten[x.{\[Theta]}\[Transpose]]

logprob[x_, \[Theta]_] := Flatten[-Log[1 + E^-a[x, \[Theta]]]]

prob[x_,\[Theta]_]:=Flatten[1/(1+E^-a[x, \[Theta]])]

cost[x_,y_,\[Theta]_,\[Lambda]_:1.0]:=
  Module[{n=Dimensions[x][[1]]},
    (Total[-y*logprob[x,\[Theta]]-(1-y)*(logprob[x,\[Theta]] - a[x,\[Theta]])]/n)+ \[Lambda]*Total[\[Theta][[2;;]]^2]/(2*n)]

grad[x_, y_, \[Theta]_, \[Lambda]_:1.0] :=
  Module[{reg = (\[Lambda]/Dimensions[x][[1]]) * Join[{0.}, \[Theta][[2;;]]]},
	Total[MapThread[#1 * #2 + reg &, {Exp[logprob[x, \[Theta]]] - y, x}]] / Dimensions[x,1][[1]]]

hessian[x_,\[Theta]_,\[Lambda]_:1.0]:=
  Module[{n=Dimensions[x][[1]],
          p=Dimensions[x][[2]]},
    Module[{reg=DiagonalMatrix[Join[{0.},ConstantArray[\[Lambda]/n,p-1]]]},
      (Total@MapThread[Times,{Exp[2*logprob[x, \[Theta]] - a[x, \[Theta]]],Map[Outer[Times,#,#]&,x]}]/n) + reg]]

update[x_, y_, \[Theta]_, \[Rho]_:.05, \[Lambda]_:1.0] :=
  Flatten[\[Theta] - \[Rho]*Inverse[hessian[x, \[Theta], \[Lambda]]].{grad[x, y, \[Theta], \[Lambda]]}\[Transpose]]

trainInternal[x_,y_,\[Lambda]_:1.0,iterations_:50]:=
  Module[{\[Theta] = RandomVariate[NormalDistribution[0,.001], Dimensions[x][[2]]],stats={}},    
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
    {"type" -> "logistic", "\[Lambda]" -> \[Lambda], "iterations" -> iterations, "\[Theta]" -> ("\[Theta]" /. result), 
     "stats" -> ("stats" /. result)}]

predict[model_, x_] :=
  If[# > Log[.5], 1., 0.]& /@ logprob[x, "\[Theta]" /. model]

End[]
EndPackage[]






