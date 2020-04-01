#include "config.h"
#include "mvc_lib.h"
#include "graph.h"
#include "nn_api.h"
#include "qnet.h"
#include "new_qnet.h"
#include "nstep_replay_mem.h"
#include "simulator.h"
#include "mvc_env.h"
#include <Eigen/Dense>
#include <mkl.h>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <string.h>

using namespace gnn;

void intHandler(int dummy) {
    exit(0);
}

int LoadModel(const char* filename)
{
    ASSERT(net, "please init the lib before use");    
    net->model.Load(filename);
    return 0;
}

int SaveModel(const char* filename)
{
    ASSERT(net, "please init the lib before use");
    net->model.Save(filename);
    return 0;
}

std::vector< std::vector<double>* > list_pred;
MvcEnv* test_env;
int Init(const int argc, const char** argv)
{
    signal(SIGINT, intHandler);
    
    cfg::LoadParams(argc, argv);     //加载参数
    GpuHandle::Init(cfg::dev_id, 1); //

    net = new QNet(); 
    //net = new NewQNet();   
    net->BuildNet();

    NStepReplayMem::Init(cfg::mem_size);
    
    Simulator::Init(cfg::num_env);
    for (int i = 0; i < cfg::num_env; ++i)
        Simulator::env_list[i] = new MvcEnv(cfg::max_n); //env_list存放的是Mvc类的数据，包括奖励，终止条件等
    test_env = new MvcEnv(cfg::max_n);

    list_pred.resize(cfg::batch_size);                   //存放的是batch_size个的指向预测节点vector的指针
    for (int i = 0; i < cfg::batch_size; ++i)
        list_pred[i] = new std::vector<double>(cfg::max_n + 10);
    return 0;
}

int UpdateSnapshot()
{
    net->old_model.DeepCopyFrom(net->model);
    return 0;
}

int InsertGraph(bool isTest, const int g_id, const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to)
{
    auto g = std::make_shared<Graph>(num_nodes, num_edges, edges_from, edges_to);
    //getchar();
    if (isTest)
        GSetTest.InsertGraph(g_id, g);
    else
        GSetTrain.InsertGraph(g_id, g);
    return 0;
}

int ClearTrainGraphs()
{
    GSetTrain.graph_pool.clear();
    return 0;
}

int PlayGame(const int n_traj, const double eps)
{
    //getchar();
    //std::cout<<"---测试-----"<<std::endl;
    //printf("Hello World！");
    Simulator::run_simulator(n_traj, eps);
    //std::cout<<"---run_simulator---成功"<<std::endl;
    return 0;
}

ReplaySample sample;
std::vector<double> list_target;
double Fit()
{
    NStepReplayMem::Sampling(cfg::batch_size, sample);
    bool ness = false;
    for (int i = 0; i < cfg::batch_size; ++i)
        if (!sample.list_term[i])
        {
            ness = true;
            break;
        }
    if (ness)
        PredictWithSnapshot(sample.g_list, sample.list_s_primes, list_pred);
    
    list_target.resize(cfg::batch_size);

    for (int i = 0; i < cfg::batch_size; ++i)
    {
        double q_rhs = 0;
        if (!sample.list_term[i])
            q_rhs = max(sample.g_list[i]->num_nodes, list_pred[i]->data());
        q_rhs += sample.list_rt[i];
        list_target[i] = q_rhs;
    }

    return Fit(sample.g_list, sample.list_st, sample.list_at, list_target);
}

double Test(const int gid, int iters)
{
    std::vector< std::shared_ptr<Graph> > g_list(1);
    std::vector< std::vector<int>* > states(1);

    test_env->s0(GSetTest.Get(gid));
    states[0] = &(test_env->action_list);
    g_list[0] = test_env->graph;

    
    double v = 0;
    int new_action;
    while (!test_env->isTerminal())
    {
        Predict(g_list, states, list_pred);
        auto& scores = *(list_pred[0]);
        new_action = arg_max(test_env->graph->num_nodes, scores.data());
        //if (scores[new_action] < 0)
          //  break;
	std::cout<<new_action<<" ";
        v += test_env->step(new_action);
    }
    char path[100];
    //sprintf(path,"/home/chenanqi/graph_comb_opt/data/karate/%d.txt",iters);
    //std::ofstream out(path,std::ios::app);
   // for(auto i:test_env->covered_set){
	//out<<i<<" ";
     //   std::cout<<i<<" ";
   // }
    //out<<"\n";
    return v;
    /*
    double cost = 0;
    int new_action;
    while (!test_env->isTerminal())
    {
        cost++;

        Predict(g_list, states, list_pred);
        new_action = arg_max(test_env->graph->num_nodes, list_pred[0]->data());
        test_env->step(new_action);
    }
    return cost;*/
}

double GetSol(const int gid, int* sol)
{
    std::vector< std::shared_ptr<Graph> > g_list(1);
    std::vector< std::vector<int>* > states(1);

    test_env->s0(GSetTest.Get(gid));
    states[0] = &(test_env->action_list);
    g_list[0] = test_env->graph;

    double cost = 0;
    int new_action;
    int len = 0;
    while (!test_env->isTerminal())
    {
        cost++;
        Predict(g_list, states, list_pred);
        new_action = arg_max(test_env->graph->num_nodes, list_pred[0]->data());
        test_env->step(new_action);
        len++;
        sol[len] = new_action;
    }
    sol[0] = len;
    return cost;    
}


int main(){
    std::cout<<"------test------"<<std::endl;
    /*
    float *A;
    A = (float *)mkl_malloc(5 * 1 * sizeof(float), 64);
    for (int i = 0; i < 5; i++){
        A[i] = i;
        std::cout<<A[i]<<std::endl;
    }*/

    
    int num_subjs = 1000;
    bool isTest = 1;
    bool isTrain = 0;
    int g_ids_test = 100, g_ids_train = 402;
    int num_nodes = 34;

    //初始化 
    const char* arr[] = {
		"-n_step","2",\
        "-avg_global","0",\
		"-dev_id","0",\
		"-min_n","34",\
		"-max_n","34",\
		"-num_env","10",\
		"-max_iter","1000000",\
		"-mem_size","500000",\
		"-g_type","barabasi_albert",\
		"-learning_rate","0.0001",\
		"-max_bp_iter","3",\
		"-net_type","QNet",\
		"-max_iter","200000",\
		"-save_dir","results/dqn-barabasi_albert/embed-64-nbp-5-rh-64",\
		"-embed_dim","64",\
		"-batch_size","64",\
		"-reg_hidden","64",\
		"-momentum","0.9",\
		"-l2","0.00",\
		"-w_scale","0.01"
	};
    int len_arr = 38;
    Init(len_arr,arr);

    //502份数据 存储在datas中 subs, edges, nodes
    std::vector<std::vector<std::vector<int>>> datas, datas_test, datas_train;
    std::vector<int> line;
	std::vector<std::vector<int>> data;
    std::ifstream filename("/home/chenanqi/graph_comb_opt/data/karate/filename_karate.txt");
    
    if (filename) {
		std::string file;
		while (getline(filename, file)) {
            file.erase(file.size()-1,1);
			std::ifstream input(file);
			if (input) {
                //std::string tmpe;
                //getline(input,tmpe);
                //std::cout<<tmpe<<std::endl;
				//line.resize(2);
				int value;
				int flag = 0;
				while (input >> value)
				{
					line.push_back(value);
                   // std::cout<<"---load-data---"<<std::endl;
					++flag;
					if (flag == 2) {
						data.push_back(line);
						line.clear();
						flag = 0;
					}
				}
				datas.push_back(data);
				data.clear();
			}
		}
	}
    std::ifstream test_filename("/home/chenanqi/graph_comb_opt/data/karate/test_filename.txt");
    
    if (test_filename) {
		std::string file;
		while (getline(test_filename, file)) {
            file.erase(file.size()-1,1);
			std::ifstream input(file);
			if (input) {
                //std::string tmpe;
                //getline(input,tmpe);
                //std::cout<<tmpe<<std::endl;
				//line.resize(2);
				int value;
				int flag = 0;
				while (input >> value)
				{
					line.push_back(value);
                    //std::cout<<"---load-data---"<<std::endl;
					++flag;
					if (flag == 2) {
						data.push_back(line);
						line.clear();
						flag = 0;
					}
				}
				datas_test.push_back(data);
				data.clear();
			}
		}
	}


    
    int n_valid = 1;
    int n_train = 1000;
    int num_edges = 0;
    int all_subjects = 500;
	std::vector<int> edges_from;
    std::vector<int> edges_to;
    std::vector<int> valid_index;
    std::vector<int> train_index;

    for(int i=0;i<n_valid;++i){
        valid_index.push_back(i);
    }
    /*
    for(int i=0;i<500;++i){
	if(std::find(valid_index.begin(),valid_index.end(),i)!=valid_index.end()){
	    datas_test.push_back(datas[i]);
	}
	else{
	    datas_train.push_back(datas[i]);
	}
    }*/
    
    //准备验证集数据
    for(int i=0;i<n_valid;++i){
        num_edges = datas_test[i].size();
        for(int j=0;j<num_edges;++j){
            edges_from.push_back(datas_test[i][j][0]-1);
            edges_to.push_back(datas_test[i][j][1]-1);
        }
        InsertGraph(isTest,i,num_nodes,num_edges,&edges_from[0],&edges_to[0]);
        edges_from.clear();
        edges_to.clear();   
    }

    //准备训练集数据
    int ngraph_train = 0;
    for(int i=0;i<n_train;++i){
        train_index.push_back(rand()%(all_subjects));
    }
    for(int i=0;i<n_train;++i){
        num_edges = datas[train_index[i]].size();
        for(int j=0;j<num_edges;++j){
            edges_from.push_back(datas[train_index[i]][j][0]-1);
            edges_to.push_back(datas[train_index[i]][j][1]-1);
        }
        InsertGraph(isTrain,ngraph_train,num_nodes,num_edges,&edges_from[0],&edges_to[0]);
        ++ngraph_train;
        edges_from.clear();
        edges_to.clear();   
    }
    //std::cout<<GSetTrain.Get(0)->adj_Matrix<<std::endl;
    //std::cout<<GSetTrain.Get(1)->adj_Matrix<<std::endl;
    /*
    :function InsertGraph: 用来给测试集和训练集添加图
    :param:
    :bool isTest:真表示为测试集添加图；假为训练集
    :const int g_id:图的索引值
    :const int num_nodes:节点数
    :const int num_edges:边数
    :const int* edges_from:目测是一个存放node的数组
    :const int* edges_to:  
    */

    for(int i=0;i<1;++i){
        PlayGame(100,1.0);
    }
    UpdateSnapshot();


    double eps_start = 1.0;
    double eps_end = 0.05;
    double eps_step = 10000.0;
    double eps = 0.0;
    double frac = 0.0;
    char model_path[100];
    double min_eigen = -100;
    int min_iter = 0;
    //std::string model_path = "/home/chenanqi/tmp/data/";
    char tmp_name[200];
    for(int iter=0;iter<1000000;++iter){
        if((iter) && (iter%5000==0)){
            train_index.clear();
            for(int i=0;i<n_train;++i){
                train_index.push_back(rand()%(all_subjects));
            }
            for(int i=0;i<n_train;++i){
                num_edges = datas[train_index[i]].size();
                for(int j=0;j<num_edges;++j){
                    edges_from.push_back(datas[train_index[i]][j][0]-1);
                    edges_to.push_back(datas[train_index[i]][j][1]-1);
                }
                InsertGraph(isTrain,ngraph_train,num_nodes,num_edges,&edges_from[0],&edges_to[0]);
                ++ngraph_train;
                edges_from.clear();
                edges_to.clear();   
            }
        }
        eps = eps_end + std::max(0.0, (eps_start - eps_end) * (eps_step - iter) / eps_step);
        if(iter%10 == 0){
            PlayGame(10,eps);
        }
	//int min_eigen=-100;
	//int min_iter=0;
        if(iter%300 == 0){
            frac = 0.0;
            
            for(int idx=0;idx<n_valid;++idx){
                frac += Test(idx,iter);
                std::cout<<std::endl;
            }
            std::cout<<"iter:"<<iter<<"eps:"<<eps<<"average size of vc:"<<(frac/n_valid)<<std::endl;    
            //sprintf(model_path,"/home/chenanqi/graph_comb_opt/data/batch128/model64/%d_iters.model",iter);
            //SaveModel(model_path);
	    if((frac/n_valid)>min_eigen){
	    	min_eigen = (frac/n_valid);
		    min_iter = iter;
		//std::cout<<min_iter<<std::endl;
	    }
	    std::cout<<min_iter<<std::endl;
        }

        if(iter%1000==0){
            UpdateSnapshot();
        }
        Fit();
    }



    return 0;
}