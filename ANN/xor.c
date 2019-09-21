#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "bp.h"

double xor[4][2]={{1,1},{0,1},{1,0},{0,0}},
       xor_ans[4][1]={{0},{1},{1},{0}};
    
int main()
{
   TrainData *td;
   NN *nn;
   int i,nodeAmount[]={2,1};
  
   nn=setUpNet(2,2,nodeAmount);
   td=setUpTD(4,(double *)xor,(double *)xor_ans,nn);
   printNet(nn);
   trainNet(500000,nn,td);
   printNet(nn);
   testNet(nn,td);
   deleteNet(nn);
   deleteTD(td);
}