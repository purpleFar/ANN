/*
2017/6/29
writing by LIN KUAN CHUNG
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bp.h"

NN *setUpNet(int inpAmount,int layer,int *nodeAmount)
{
  int i,j,k,wAmount;
  NN *nn=malloc(sizeof(NN));
  
  srand(time(NULL));
  nn->network=malloc(layer*sizeof(Z *));
  nn->inpAmount=inpAmount;
  nn->nodeAmount=nodeAmount;
  nn->layer=layer;
  for(i=0;i<layer;i++)
  {
     nn->network[i]=malloc(nodeAmount[i]*sizeof(Z));
     for(j=0;j<nodeAmount[i];j++)
     {
        wAmount=(i==0)?inpAmount:nodeAmount[i-1];
        nn->network[i][j].total=malloc(wAmount*sizeof(double));
        for(k=0;k<wAmount;k++)//set the initial value(-0.99~0.99)
           nn->network[i][j].total[k]=((double)(rand()%99+1)/100);
        nn->network[i][j].b=((double)(rand()%99+1)/100);
     }
  }
  return nn; 
}

TrainData *setUpTD(int dataAmount,double *input,double *output,NN *nn)
{
   int i;
   TrainData *td=malloc(sizeof(TrainData));
   
   td->dataAmount=dataAmount;
   td->input=malloc(dataAmount*sizeof(double *));
   td->output=malloc(dataAmount*sizeof(double *));
   for(i=0;i<dataAmount;i++)
   {
      td->input[i]=input+(i*nn->inpAmount);
      td->output[i]=output+(i*nn->nodeAmount[nn->layer-1]);
   }
   return td;
}

void trainNet(int time,NN *nn,TrainData *td)
{
   int i,j,k,s,quit,wAmount;
   double **network;//store activation function(z)
   NN *bp=setUpNet(nn->inpAmount,nn->layer,nn->nodeAmount);//store dC/dz
   
   for(quit=0;quit<time;quit++)
   {
      for(i=0;i<td->dataAmount;i++)
      {
         network=product(nn,td->input[i]);
         BP(nn,td->output[i],network,bp); 
         for(j=0;j<nn->layer;j++)//update all weights
         {
            for(k=0;k<nn->nodeAmount[j];k++)
            {
               wAmount=(j==0)?nn->inpAmount:nn->nodeAmount[j-1];               
               for(s=0;s<wAmount;s++)
               {
                  if(j==0)//W=W-a(dz/dW)(dC/dz)
                     nn->network[j][k].total[s]-=ALPHA*td->input[i][s]*bp->network[j][k].total[s];
                  else
                     nn->network[j][k].total[s]-=ALPHA*network[j-1][s]*bp->network[j][k].total[s];
               }
               nn->network[j][k].b-=ALPHA*bp->network[j][k].b;
            }
         }
         if(quit==0&&i==0)
         {
            printf("--------------\nbp:\n");
            printNet(bp);
            printf("-----------------\n");
            printNet(nn);
            testNet(nn,td);
            printf("----------------\n");
         }
      }
   }
   
   free(bp);
   free(network);
}

void BP(NN *nn,double *output,double **network,NN *bp)
{
   int i,j,k,s,wAmount;
   for(i=nn->layer-1;i>=0;i--)
   {
      if(i==nn->layer-1)
      {
         for(j=0;j<nn->nodeAmount[i];j++)
         {
            for(k=0;k<nn->nodeAmount[i-1];k++)//loss function: C(theta)=1/2(Ans-y)^2
               bp->network[i][j].b=bp->network[i][j].total[k]=(network[i][j]<=0)?0:(network[i][j]-output[j]);//dC/dz=(dC/dy)(dy/dz)=(y-Ans)ReLU(z)
         }
      }
      else
      {
         for(j=0;j<nn->nodeAmount[i];j++)
         {
            wAmount=(i==0)?nn->inpAmount:nn->nodeAmount[i-1];
            for(k=0;k<wAmount;k++)
            {
               bp->network[i][j].b=bp->network[i][j].total[k]=0;
               if(network[i][j]>0)
               {   
                  for(s=0;s<nn->nodeAmount[i+1];s++)
                     bp->network[i][j].total[k]+=nn->network[i+1][s].total[j]*bp->network[i+1][s].total[j];
                  bp->network[i][j].b=bp->network[i][j].total[k];  //dC/dz=(w'*dC/dz'+w"*dC/dz"+...)ReLU(z)
               }
            }
         }
      }
   }
}

double **product(NN *nn,double *input)
{
   int i,j,k;
   double **network=malloc(nn->layer*sizeof(double *));
   
   for(i=0;i<nn->layer;i++)
   {
      network[i]=malloc(nn->nodeAmount[i]*sizeof(double));
      if(i==0)
         funcZ("ReLU",nn->inpAmount,nn->nodeAmount[i],nn->network[0],input,network[0]);
      else
         funcZ("ReLU",nn->nodeAmount[i-1],nn->nodeAmount[i],nn->network[i],network[i-1],network[i]);
   }
   return network;
}

void funcZ(char *type,int lastLayer,int thisLayer,Z *weights,double *input,double *output)
{
   int i,j;
   for(i=0;i<thisLayer;i++)
   {
      for(j=0;j<lastLayer;j++)
         output[i]+=input[j]*weights[i].total[j];
      output[i]+=weights[i].b;
      if(strcmp(type,"ReLU")==0)
         output[i]=(output[i]<0)?0:output[i];//activation function
   }
}

void printNet(NN *nn)
{
   int i,j,k,wAmount;
   
   printf("layer=%d,inpAmount=%d\n",nn->layer,nn->inpAmount);
   for(i=0;i<nn->layer;i++)
   {
      printf("nodeAmount=%d\n",nn->nodeAmount[i]);
      for(j=0;j<nn->nodeAmount[i];j++)
      {
         wAmount=(i==0)?nn->inpAmount:nn->nodeAmount[i-1];
         for(k=0;k<wAmount;k++)
            printf("w%d=%lf, ",k+1,nn->network[i][j].total[k]);
         printf("b=%lf\n",nn->network[i][j].b);
      }
   }
}

void testNet(NN *nn,TrainData *td)
{
   int i,j;
   double **result;
   
   printf("Data:\n");
   for(i=0;i<td->dataAmount;i++)
   {
      printf("input:(");
      for(j=0;j<nn->inpAmount-1;j++)
         printf("%lf,",td->input[i][j]);
      printf("%lf)   ",td->input[i][nn->inpAmount-1]);
      printf("output:(");
      for(j=0;j<nn->nodeAmount[nn->layer-1]-1;j++)
         printf("%lf,",td->output[i][j]);
      printf("%lf)\n",td->output[i][nn->nodeAmount[nn->layer-1]-1]);
   }
   

   printf("Test:\n");
   for(i=0;i<td->dataAmount;i++)
   {
      printf("input:(");
      for(j=0;j<nn->inpAmount-1;j++)
         printf("%lf,",td->input[i][j]);
      printf("%lf)   ",td->input[i][nn->inpAmount-1]);
      result=product(nn,td->input[i]);
      printf("output:(");
      for(j=0;j<nn->nodeAmount[nn->layer-1]-1;j++)
         printf("%lf,",result[nn->layer-1][j]);
      printf("%lf)\n",result[nn->layer-1][nn->nodeAmount[nn->layer-1]-1]);
   }
   
   free(result);
}

void deleteNet(NN *nn)
{
   int i,j;

   for(i=0;i<nn->layer;i++)
   {
      for(j=0;j<nn->nodeAmount[i];j++)
         free(nn->network[i][j].total);
      free(nn->network[i]);
   }
   free(nn->network);
   free(nn);
}

void deleteTD(TrainData *td)
{
   free(td->input);
   free(td->output);
   free(td);
}
