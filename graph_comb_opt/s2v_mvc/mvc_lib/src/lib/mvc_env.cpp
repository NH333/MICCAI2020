#include "mvc_env.h"
#include "graph.h"
//#include <mkl.h>
#include <cassert>
#include <random>
#include <algorithm>
#include <vector>

MvcEnv::MvcEnv(double _norm) : IEnv(_norm)
{

}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
    sumEigenValues = 0;
    //numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double MvcEnv::step(int a)
{
    assert(graph);
    assert(covered_set.count(a) == 0);

    state_seq.push_back(action_list);
    act_seq.push_back(a);

    covered_set.insert(a);
    action_list.push_back(a);

    /*对加入的a，在矩阵中a行a列置零，求特征值*/
    MatrixXd tmp_W = graph->adj_Matrix;
    MatrixXd D;
	MatrixXd L;
    MatrixXd tmp_;
    MatrixXd eigenVal;
    std::vector<double> eigen_values;
    double old_sumEigenValues = sumEigenValues;
    eigen_values.clear();

    D.setZero(148, 148);
    L.setZero(148, 148);
    tmp_.setZero(148,1);

    for(auto covered:covered_set){
        for(int i=0;i<148;++i){
            tmp_W(i,covered) = 0;
            tmp_W(covered,i) = 0;
        }
    }
    for(int i=0;i<148;++i){
        tmp_W(i,a) = 0;
        tmp_W(a,i) = 0;
    }
    tmp_ = tmp_W.rowwise().sum();
    for (int i = 0; i < 148; ++i) {
		D(i, i) = tmp_(i,0);
	}
    L = D - tmp_W;
    eigenVal = L.eigenvalues().real();
	for (int i = 0; i < 148; ++i) {
		eigen_values.push_back(eigenVal(i, 0));
	}
    std::sort(eigen_values.begin(),eigen_values.end());
    sumEigenValues = 0;
    for(int i=0;i<148;++i){
        sumEigenValues += eigen_values[i];
    }






    // for (auto& neigh : graph->adj_list[a])
    //     if (covered_set.count(neigh) == 0)
    //         numCoveredEdges++;
    

    double r_t = getReward(old_sumEigenValues);
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}

int MvcEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (covered_set.count(i) == 0)
            avail_list.push_back(i);
    // for (int i = 0; i < graph->num_nodes; ++i)
    //     if (covered_set.count(i) == 0)
    //     {
    //         bool useful = false;
    //         for (auto& neigh : graph->adj_list[i])
    //             if (covered_set.count(neigh) == 0)
    //             {
    //                 useful = true;
    //                 break;
    //             }
    //         if (useful)
    //             avail_list.push_back(i);
    //     }
    
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool MvcEnv::isTerminal()
{
    assert(graph);
    return((int)covered_set.size()==8);
    //return graph->num_edges == numCoveredEdges;
}

double MvcEnv::getReward(double old_sumEigenValues)
{
    //return -1.0 / norm;
    //return -(sumEigenValues-old_sumEigenValues) / 34.0;
        return -(sumEigenValues-old_sumEigenValues)/148.0;
}
