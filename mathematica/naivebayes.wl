(* ::Package:: *)

BeginPackage["mimir`naivebayes`"];

Begin["`Private`"];

message::onehoterror = "y is not one hot encoded";

train[x_, y_, \[Alpha]_:1] :=
  train[x, y, "multinomial", \[Alpha]]

train[x_, y_, "bernoulli", \[Alpha]_:1] :=
  Module[{classes, classCounts, featureCounts, classLogProb, featureLogProb},
    If[!mimir`data`checkOneHotArray[y], 
      Message[message::onehoterror];
      Return[$Failed]];
    classes = Range@Dimensions[y][[2]];
    featureCounts = y\[Transpose].x;
    classCounts = Total[y];
    featureLogProb = (Log[# + \[Alpha]] - Log[classCounts + 2*\[Alpha]] & /@ (featureCounts\[Transpose]))\[Transpose];
    classLogProb = Log@classCounts - Log@Total@classCounts;
    {"\[Alpha]" -> \[Alpha], "classes" -> classes, "featureCounts" -> featureCounts,
     "classCounts" -> classCounts, "featureLogProb" -> featureLogProb,
     "classLogProb" -> classLogProb, "type" -> "bernoulli"}]
    
train[x_, y_, "multinomial", \[Alpha]_:1] :=
  Module[{classes, priorCounts, classCounts, featureCounts, classLogProb, featureLogProb},
    If[!mimir`data`checkOneHotArray[y], 
      Message[message::onehoterror];
      Return[$Failed]];
    classes = Range@Dimensions[y][[2]];
    featureCounts =y\[Transpose].x + \[Alpha];
    classCounts = Total[featureCounts, {2}];
    priorCounts = Total[y, {1}];
    featureLogProb = MapIndexed[Log[#1] - Log[classCounts[[#2]][[1]]] &, featureCounts];
    classLogProb = Log@priorCounts - Log@Total@priorCounts;
    {"\[Alpha]" ->\[Alpha] , "classes" -> classes, "featureCounts" -> featureCounts,
     "classCounts" -> classCounts, "featureLogProb" -> featureLogProb,
     "classLogProb" -> classLogProb, "type" -> "multinomial"}]

train[x_, y_, "gaussian"] :=
  Module[{classes, priorCounts, classLogProb, mu, sigma},
    classes = Range@Dimensions[y][[2]];
    priorCounts = Total[y, {1}];
    classLogProb = Log@priorCounts - Log@Total@priorCounts;
    mu = Mean@x;
    sigma = Variance@x + 1.^-9;
    {"classes" -> classes, "classLogProb" -> classLogProb, "mu" -> mu, "sigma" -> sigma,
     "type" -> "gaussian"}]

logprob[m_, x_] :=
  logprob[m, x, "type" /. m]

logprob[m_, x_, "bernoulli"] :=
  Module[{featProb = "featureLogProb" /. m, negFeatProb, classProb= "classLogProb" /. m},
    negFeatProb = Log[1 - Exp[featProb]];
    (* jointLik=Map[# +c p &, xBin.featProb\[Transpose] + (1 - xBin).negFeatProb\[Transpose]] *)
    (* packed formulation from sklearn *)
    Map[# + classProb + Total[negFeatProb, {2}] &, x.(featProb - negFeatProb)\[Transpose]]]

logprob[m_, x_, "multinomial"] :=
  Module[{featProb = "featureLogProb" /. m, classProb = "classLogProb" /. m},
    Map[# + classProb&,(x.featProb\[Transpose])]]

logprob[m_, x_, "gaussian"] :=
  Module[{mu = "mu" /. m, sigma = "sigma" /. m, classProb = "classLogProb" /. m,
          classes= "classes" /. m},
    Table[Map[Total[classProb[[#]] - 1/(2*sigma[[#]]) (x[[i]]-mu[[#]])^2 -1/2 Log[2*\[Pi]*sigma[[#]]]] &,
      classes] // N, {i,1,Length@x}]]

predict[m_, x_] :=
  Flatten[Position[#, Max[#]] & /@ logprob[m, x]] - 1

prob[m_, x_] :=
  Exp[logprob[m, x]]

End[];
EndPackage[];
