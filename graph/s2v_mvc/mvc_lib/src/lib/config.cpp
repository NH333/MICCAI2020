#include "config.h"

int cfg::max_bp_iter = 1;
int cfg::embed_dim = 64;
int cfg::dev_id = 0;
int cfg::batch_size = 32;
int cfg::max_iter = 1000000;
int cfg::reg_hidden = 32;
int cfg::node_dim = 0;
int cfg::aux_dim = 0;
int cfg::min_n = 148;
int cfg::max_n = 148;
int cfg::mem_size = 500000;
int cfg::num_env = 10;
int cfg::n_step = 2;
int cfg::edge_dim = 4;
int cfg::avg_global = -1;
int cfg::edge_embed_dim = -1;
Dtype cfg::learning_rate = 0.0005;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0.9;
Dtype cfg::w_scale = 0.01;
const char* cfg::save_dir = "./saved";



