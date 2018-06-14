(* ::Package:: *)

BeginPackage["mimir`text`"];

makeVocab::usage = "";
vectorize::usage = "";

Begin["`Private`"];

makeVocab[docs_, maxDf_:.5, minDf_:5] :=
  Module[{
    n = Length@docs,
    counts = Fold[Module[{words=ToLowerCase/@TextCases[#2,"Word"]},
      {Merge[{#1[[1]],Counts@words},Total],
       Merge[{#1[[2]],Counts@DeleteDuplicates@words},Total]}]&,
      {<||>,<||>}, docs]},
    Module[{index = Map[If[counts[[2]][[#]] >= minDf || counts[[2]][[#]] <= maxDf*n,#]&, Keys[counts[[2]]]]},
      {"index2Word" -> index,
       "word2Index" -> PositionIndex[index]}]]

vectorize[docs_, vocab_] :=
  Module[{n = Length@docs, p = Length@("word2Index" /. vocab)},
    Module[{m = SparseArray[{}, {n, p}]},
      For[i = 1, i <= n, i++,
        KeyValueMap[(m[[i, #1]] = #2) &,
          Counts@Flatten@DeleteCases[("word2Index" /. vocab)[[#]] & 
            /@ ToLowerCase /@ TextCases[docs[[i]], "Word"], _Missing]]];
      m]]

End[];
EndPackage[];
