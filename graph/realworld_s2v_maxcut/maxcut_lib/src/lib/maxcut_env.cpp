#include "maxcut_env.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>

MaxcutEnv::MaxcutEnv(double _norm) : IEnv(_norm)
{

}

void MaxcutEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    cut_set.clear();
    action_list.clear();
    //cutWeight = 0;
    sumEigenValues = 0;
    MatrixXd tmp_W = graph->adj_Matrix; MatrixXd D; MatrixXd L; MatrixXd tmp_; MatrixXd eigenVal;
    std::vector<double> eigen_values;
    int all_node = 116;
    eigen_values.clear();
    D.setZero(all_node, all_node);
    L.setZero(all_node, all_node);
    tmp_.setZero(all_node,1);

    tmp_ = tmp_W.rowwise().sum();
    for (int i = 0; i < all_node; ++i) {
                D(i, i) = tmp_(i,0);
        }
    L = D - tmp_W;

    eigenVal = L.eigenvalues().real();
        for (int i = 0; i < all_node; ++i) {
                eigen_values.push_back(eigenVal(i, 0));
        }
    for(int i=0;i<all_node;++i){
        sumEigenValues += eigen_values[i];
    }
    //sumEigenValues = 0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double MaxcutEnv::step(int a)
{
    assert(graph);
    assert(cut_set.count(a) == 0);
    assert(a >= 0 && a < graph->num_nodes);

    state_seq.push_back(action_list);
    act_seq.push_back(a);

    cut_set.insert(a);
    action_list.push_back(a);

    /*对加入的a，在矩阵中a行a列置零，求特征值*/
    MatrixXd tmp_W = graph->adj_Matrix;
    MatrixXd D;
    MatrixXd L;
    MatrixXd tmp_;
    MatrixXd eigenVal;
        MatrixXd::Index maxRow,minCol;
    std::vector<double> eigen_values;
    double old_sumEigenValues = sumEigenValues;
    sumEigenValues = 0;
    int all_node = 116;
    eigen_values.clear();

    D.setZero(all_node, all_node);
    L.setZero(all_node, all_node);
    tmp_.setZero(all_node,1);

    for(auto covered:cut_set){
        for(int i=0;i<all_node;++i){
            tmp_W(i,covered) = 0;
            tmp_W(covered,i) = 0;
        }
    }

    tmp_ = tmp_W.rowwise().sum();
    for (int i = 0; i < all_node; ++i) {
		D(i, i) = tmp_(i,0);
	}
    L = D - tmp_W;
    double maxL_value = tmp_W.maxCoeff(&maxRow,&minCol);
    eigenVal = L.eigenvalues().real();
	for (int i = 0; i < all_node; ++i) {
		eigen_values.push_back(eigenVal(i, 0));
	}
    //std::sort(eigen_values.begin(),eigen_values.end());
    for(int i=0;i<all_node;++i){
        sumEigenValues += eigen_values[i];
    }
    
    //std::cout<<a<<":"<<sumEigenValues<<std::endl;
    /*
    double old_cutWeight = cutWeight;
    for (auto& neigh : graph->adj_list[a])        
        if (cut_set.count(neigh.first) == 0)
            cutWeight += neigh.second;
        else
            cutWeight -= neigh.second;
    
    double r_t = getReward(old_cutWeight);
    */
    double r_t = getReward(old_sumEigenValues);
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}

int MaxcutEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (cut_set.count(i) == 0)
            avail_list.push_back(i);
    
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool MaxcutEnv::isTerminal()
{
    assert(graph);
    return((int)cut_set.size()==8);
   // return ((int)cut_set.size() + 1 >= graph->num_nodes);
}

double MaxcutEnv::getReward(double old_sumEigenValues)
{
    //return (cutWeight - old_cutWeight) / norm;
    return -(sumEigenValues-old_sumEigenValues)/116.0;
}
