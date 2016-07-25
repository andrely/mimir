(* ::Package:: *)

BeginPackage["mimir`optimizers`"]

miniBatch::usage = "";

cgdFR::usage = "Fletcher Reeves Conjugate Graduate Descent";
newton::usage = "Newtons method";
steepgd::usage = "Steepest Gradient Descent";
sgd::usage = "Stochastic Gradient Descent";
lineSearch::usage = "";

Begin["`Private`"]

sign[a_, b_] := a*b/Abs[b]

bracket[a_, b_, func_] :=
  Module[{gold = 1.618034, glimit = 100, tiny = 10^-20, ax = a, bx = b, cx},
    Module[{fa = func[a], fb = func[b], fc},
      If[fb > fa, {ax, bx} = {bx, ax}; {fb, fa} = {fa, fb}];
      cx = bx + gold*(bx - ax);
      fc = func[cx];
      While[fb > fc,
        Module[{r = (bx - ax)*(fb - fc), q = (bx - cx)*(fb - fa)},
          Module[{u = bx - ((bx - cx)*q - (bx - ax)*r) / (2.0*(sign[Max[Abs[q - r], tiny], q - r])),
                   ulim = bx + glimit*(cx - bx), fu},
          Which[
            ((bx - u)*(u - cx)) > 0.,
              (fu = func[u];
               Which[
                 fu < fc,
                 ax=bx;
                 bx=u;
                 fa=fb;
                 fb=fu;
                 Return[{ax,bx,cx,fa,fb,fc}],
                 fu>fb,
                 cx=u;
                 fc=fu;
                 Return[{ax,bx,cx,fa,fb,fc}]];
               u=cx*gold*(cx-bx);
               fu=func[u]),
            ((cx-u)*(u-ulim))>0.,
            (fu=func[u];
             If[fu<fc,
               {bx,cx,u}={cx,u,u+gold*(u-cx)};
               {fb,fc,fu}={fc,fu,func[u]}]),
            ((u-ulim)*(ulim-cx))>=0.,
            (u=ulim;
             fu=func[u]),
            True,
            (u=cx+gold*(cx-bx);
             fu=func[u])];
          {ax,bx,cx}={bx,cx,u};
          {fa,fb,fc}={fb,fc,fu}]]];
      (* make sure bracket is in ascending value order for golden section to work
         BUG? *)
      If[ax > cx,
        {cx, bx, ax, fc, fb, fa},
        {ax,bx,cx,fa,fb,fc}]]]

goldenSection[bracketing_, func_] :=
  Module[{ax, bx, cx, fa, fb, fc, r=.61803399, c, x0, x1, x2, x3, f1, f2, tol=3.0*10^-8,
          xmin, fmin, iter = 1},
    {ax, bx, cx, fa, fb, fc} = bracketing;
    c = 1. - r;
    x0 = ax;
    x3 = cx;
    If[Abs[cx - bx] > Abs[bx - ax],
      (x1 = bx;
       x2 = bx - c*(cx - bx)),
      (x2 = bx;
       x1 = bx - c*(bx - ax))];
    f1 = func[x1];
    f2 = func[x2];
    While[(Abs[x3 - x0] > tol (* *(Abs[x1]-Abs[x2]) *)) && iter < 100,
      iter++;
      If[f2 < f1,
        ({x0, x1, x2} = {x1, x2, r*x2 + c*x3};
         {f1, f2} = {f2, func[x2]}),
        ({x3, x2, x1} = {x2, x1, r*x1 + c*x0};
         {f2, f1} = {f1, func[x1]})]];
    If[f1 < f2,
      (xmin=x1;
       fmin=f1),
      (xmin=x2;
       fmin=f2)];
    {xmin,fmin}]

lineSearch[p_, r_, f_] := goldenSection[bracket[p, r, f], f][[1]]

miniBatch[x_, y_, batchSize_:10] :=
  Module[{sampleIdx = RandomSample[Range@Length@x],
          batchNum = Ceiling[Length@x / batchSize]},
    Module[{batchIdx = Table[{1+batchSize*i, Min[batchSize+batchSize*i, Length@x]}, {i, 0, batchNum-1}],
            xSample = x[[sampleIdx, All]], ySample = y[[sampleIdx]]},
      {xSample[[Apply[Range, #]]], ySample[[Apply[Range, #]]]} & /@ batchIdx]]

cgdFR[cost_, grad_, \[Theta]_] :=
  Module[{iter = 0, g = grad[\[Theta]], theta = \[Theta], gPrev, \[Beta], pPrev, \[Alpha], stats = {}},
    Module[{p = -g},
      While[Norm[g] > .000001 && iter < 10,
        iter++;
        AppendTo[stats, {iter, cost[theta]}];
        \[Alpha] = lineSearch[-1, 1., cost[theta + #*p] &];
        theta = theta + \[Alpha]*p;
        gPrev = g;
        g = grad[theta];
        \[Beta] = Flatten[g].Flatten[g] / Flatten[gPrev].Flatten[gPrev];
        pPrev = p;
        p = g + \[Beta]*pPrev];
      {"\[Theta]" -> theta, "stats" -> stats}]]

newton[cost_, grad_, hessian_, \[Theta]_, iterations_:50]:=
  Module[{stats={}, theta = \[Theta], h, g},    
    Module[{j = 0, newJ = cost[theta], iter=0},
      While[Abs[j - newJ] > .00001 && iter < iterations,
        iter++;
        AppendTo[stats, {iter, newJ}];
        j = newJ;
        g = grad[theta];
        h = hessian[theta];
        theta = theta - ArrayReshape[Inverse[h].{Flatten[g]}\[Transpose], Dimensions[theta]];
        newJ = cost[theta]];
      {"\[Theta]" -> theta, "stats" -> stats}]]

steepgd[cost_, grad_, \[Theta]_, \[Rho]_:.05, maxIter_:50]:=
  Module[{theta = \[Theta], j = 0, newJ = cost[\[Theta]], iter = 0, stats = {}},
    While[Abs[j - newJ] > .00001 && iter < maxIter,
      iter++;
      AppendTo[stats, {iter, newJ}];
      j = newJ;
      theta = theta - \[Rho] * grad[theta];
      newJ = cost[theta]];
    {"\[Theta]" -> theta, "stats" -> stats}]

sgd[cost_, grad_, batchFunc_, \[Theta]_, \[Rho]_:.05, maxIter_:50]:=
  Module[{stats = {}, theta = \[Theta], j = 0, newJ = cost[\[Theta]], iter = 0},
    While[Abs[j - newJ] > .00001 && iter < maxIter,
      iter++;
      AppendTo[stats, {iter,newJ}];
      j = newJ;
      theta = Fold[theta - \[Rho] * grad[#1, #2] &, theta, batchFunc[]];
      newJ = cost[theta]];
    {"\[Theta]" -> theta, "stats" -> stats}]

End[]
EndPackage[]



