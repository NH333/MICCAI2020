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

double Test(const int gid, int index)
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
        v += test_env->step(new_action) * 148;
    }
    char path[100];
    sprintf(path, "/home/chenanqi/graph_comb_opt/data/mvc/TestReTest/resultReTest/%d.txt",index);
    std::ofstream out(path, std::ios::app);
    for(auto i:test_env->covered_set){
        out<<i<<" ";
       // std::cout<<i<<" ";
    }
    out<<"\n";
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
    const char* arr[] = {
		"-n_step","1",\
        "-avg_global","0",\
		"-dev_id","0",\
		"-min_n","148",\
		"-max_n","148",\
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
		"-batch_size","128",\
		"-reg_hidden","32",\
		"-momentum","0.9",\
		"-l2","0.00",\
		"-w_scale","0.01"
	};
    int len_arr = 38;
    Init(len_arr,arr);
    std::vector<std::vector<std::vector<int>>> retest_datas;
    std::vector<int> line;
	std::vector<std::vector<int>> data;
    std::ifstream retest_filename("/home/chenanqi/graph_comb_opt/data/mvc/TestReTest/ReTest/retest_filename.txt");
    
    if (retest_filename) {
		std::string file;
		while (getline(retest_filename, file)) {
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
				retest_datas.push_back(data);
				data.clear();
			}
		}
	}

    //char* model_file = "/home/chenanqi/graph_comb_opt/data/batch128/model16/29400_iters.model";
    int n_valid = 119;
    int num_edges = 0;
    bool isTest = 1;
    int num_nodes = 148;
    
	std::vector<int> edges_from;
    std::vector<int> edges_to;
    std::vector<int> valid_index;
    std::vector<int> train_index;


    //LoadModel(model_file);

    for(int i=0;i<n_valid;++i){
        valid_index.push_back(i);
    }
    //准备测试集数据
    for(int i=0;i<n_valid;++i){
        num_edges = retest_datas[valid_index[i]].size();
        for(int j=0;j<num_edges;++j){
            edges_from.push_back(retest_datas[valid_index[i]][j][0]-1);
            edges_to.push_back(retest_datas[valid_index[i]][j][1]-1);
        }
        InsertGraph(isTest,i,num_nodes,num_edges,&edges_from[0],&edges_to[0]);
        edges_from.clear();
        edges_to.clear();   
    }

    
    std::ifstream retest_model_filename("/home/chenanqi/graph_comb_opt/data/mvc/TestReTest/Model/retest_model_filename_emb64.txt");
    if (retest_model_filename) {
                std::string file;
        int index_model = 0;
		while (getline(retest_model_filename, file)){

            file.erase(file.size()-1,1);
            LoadModel(file.c_str());
            double frac = 0.0;
            for(int idx=0;idx<n_valid;++idx){
        
                frac += Test(idx,index_model);
               // std::cout<<std::endl;
            }
            index_model += 300;
            std::cout<<"average EigenValues:"<<(frac/n_valid)<<std::endl;
        }
	}

    return 0; 
}
