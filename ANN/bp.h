#define ALPHA 0.03

typedef struct
{
  double *total;
  double b;
}Z;

typedef struct
{
   Z **network;
   int layer;
   int *nodeAmount;
   int inpAmount;
}NN;

typedef struct
{
   double **input,**output;
   int dataAmount;
}TrainData;

NN *setUpNet(int inpAmount,int layer,int *nodeAmount);
TrainData *setUpTD(int dataAmount,double *input,double *output,NN *nn);
void trainNet(int time,NN *nn,TrainData *td);
void BP(NN *nn,double *output,double **network,NN *bp);
double **product(NN *nn,double *input);
void funcZ(char *type,int lastLayer,int thisLayer,Z *weights,double *input,double *output);
void printNet(NN *nn);
void testNet(NN *nn,TrainData *td);
void deleteNet(NN *nn);
void deleteTD(TrainData *td);