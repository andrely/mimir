(* ::Package:: *)

BeginPackage["mimir`optimizers`"];

miniBatch::usage = "";

bfgs::usage = "Quasinewton BFGS";
lgfgs::usage = "Quasinewton low memory BFGS";
cgdFR::usage = "Fletcher Reeves Conjugate Graduate Descent";
newton::usage = "Newtons method";
steepgd::usage = "Steepest Gradient Descent";
sgd::usage = "Stochastic Gradient Descent";
lineSearch::usage = "";

Begin["`Private`"];

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
               u=cx + gold*(cx-bx);
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
        {ax, bx, cx, fa, fb, fc}]]]

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
    {xmin, fmin}]

lineSearch[p_, r_, f_] := goldenSection[bracket[p, r, f], f][[1]]

miniBatch[x_, y_, batchSize_:10] :=
  Module[{sampleIdx = RandomSample[Range@Length@x],
          batchNum = Ceiling[Length@x / batchSize]},
    Module[{batchIdx = Table[{1+batchSize*i, Min[batchSize+batchSize*i, Length@x]}, {i, 0, batchNum-1}],
            xSample = x[[sampleIdx, All]], ySample = y[[sampleIdx]]},
      {xSample[[Apply[Range, #]]], ySample[[Apply[Range, #]]]} & /@ batchIdx]]

lbfgsIter[theta_, cost_, grad_, sk_, yk_] :=
  Module[{g = grad[theta], gamma, alphas = {}, rhos = {}, h0, q, r, p, 
          alpha, thetanew, gnew},
    If[Length@sk == 0, gamma = theta.g/g.g, gamma = sk[[-1]].yk[[-1]]/yk[[-1]].yk[[-1]]];
    q=g;
    Module[{i, \[Rho], \[Alpha]},
      For[i = 1, i <= Length@sk, i++,
        \[Rho] = 1/yk[[i]].sk[[i]];
        \[Alpha] = \[Rho]*sk[[i]]*q;
        q = q - \[Alpha]*yk[[i]];
        AppendTo[alphas, \[Alpha]];
        AppendTo[rhos, \[Rho]]]];
    r = gamma*q;
    Module[{i, \[Beta]},
      For[i = 1, i <= Length@sk, i++,
        \[Beta] = rhos[[i]]*yk[[i]].r;
        r = r + sk[[i]](alphas[[i]] - \[Beta])]];
    p = r;
    alpha = lineSearch[-1, 1., cost[theta + #*p] &];
    thetanew = theta + alpha*p;
    gnew = grad[thetanew];
    {thetanew,gnew}]
    
lbfgs[cost_, grad_, \[Theta]_]:=
  Module[{iter = 0, g = grad[\[Theta]], stats = {}, theta = \[Theta], sk = {}, yk = {},
          thetanew, gnew, c = 1, cnew = 0},
    While[Norm[g]>.00001&&c-cnew>0&&iter<50,
      c=cost[theta];
      {thetanew, gnew} = lbfgsIter[theta, cost,grad, sk,yk];
      cnew = cost[thetanew];
      PrependTo[sk, thetanew - theta];
      PrependTo[yk, gnew - g];
      theta = thetanew;
      g = gnew;
      sk[[1 ;; Min[30, Length@sk]]];
      yk[[1 ;; Min[30, Length@yk]]];
      iter++;
      AppendTo[stats, {iter, cost[theta], If[Length@yk == 0, Norm[g], Norm[yk[[1]]]]}];];
    {"\[Theta]" -> theta, "stats" -> stats}]

bfgsIter[x_, c_, g_, h_]:=
  Module[{df = g[x], p, \[Alpha], s, newx, newdf, yk, newh},
    p = LinearSolve[h, df];
    \[Alpha] = lineSearch[-1, 1. , c[x + #*p] &];
    s= \[Alpha]*p;
    newx = x + s;
    newdf = g[newx];
    yk = newdf - df;
    newh = h + Outer[Times, yk, yk]/yk.s - h.Outer[Times, s, s].h/s.h.s;
    {newx, newdf, newh}]
    
bfgs[cost_, grad_, \[Theta]_]:=
  Module[{iter = 0, g = grad[\[Theta]], stats = {}, theta = \[Theta], h = IdentityMatrix[Length@\[Theta]]},
    While[Norm[g] > .000001 && iter < 50,
      iter++;
      AppendTo[stats, {iter, cost[theta], Norm[g]}];
      {theta, g, h} = bfgsIter[theta, cost, grad, h]];
    {"\[Theta]" -> theta, "stats" -> stats}]

cgdFR[cost_, grad_, \[Theta]_] :=
  Module[{iter = 0, g = grad[\[Theta]], theta = \[Theta], gPrev, \[Beta], pPrev, \[Alpha], stats = {}},
    Module[{p = -g},
      While[Norm[g] > .000001 && iter < 50,
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

End[];
EndPackage[];
