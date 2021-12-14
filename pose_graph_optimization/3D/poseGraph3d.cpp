#include <iostream>
#include <string>
#include <fstream>

#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;

#include<ceres/ceres.h>
#include<Eigen/Dense>

//Reference : https://github.com/mpkuse/toy-pose-graph-optimization-ceres/blob/master/ceres_vertigo.cpp

//Store types of edges
enum EdgeType{
    ODOMETRY_EDGE = 0,
    CLOSURE_EDGE,
};

struct Pose3d
{ 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<double, 3,1> pos;  // xyz 
    Eigen::Quaternion<double> quat; // w q
    int index;

    Pose3d(const Pose3d& _pose3d){
        this->index = _pose3d.index;
        this->pos = _pose3d.pos;
        this->quat = _pose3d.quat;
    }

    Pose3d(int index, vector<double>& pos, vector<double>& quat):
        index(index)
        {
            this->pos.x() = pos[0];
            this->pos.y() = pos[1];
            this->pos.z() = pos[2];

            this->quat.x() = quat[0];
            this->quat.y() = quat[1];
            this->quat.z() = quat[2];
            this->quat.w() = quat[3];
        }
};

class Edge
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeType edge_type;
    Pose3d *a, *b;
    Eigen::Matrix<double, 3,1> dpos;  // xyz 
    Eigen::Quaternion<double> dquat; // w q

    Edge(Pose3d* a, Pose3d *b, EdgeType edge_type):
    a(a), b(b), edge_type(edge_type){}

    void setEdgeTransform(vector<double> &dpos, vector<double> &dquat){
        this->dpos.x() = dpos[0];
        this->dpos.y() = dpos[1];
        this->dpos.z() = dpos[2];
    
        this->dquat.x() = dquat[0];
        this->dquat.y() = dquat[1];
        this->dquat.z() = dquat[2];
        this->dquat.w() = dquat[3];
    }
};

// Read G20 format, Write nodes to a file.
class ReadG20
{
    public:
        vector<Pose3d*> nodes;
        vector<Edge*> odometry_edges;
        vector<Edge*> closure_edges;

        ReadG20(const string &fName){
            fstream fp;
            fp.open(fName.c_str(), ios::in);

            string line;
            int vertices;
            int edges;

            vector<double> pos(3, 0.0);
            vector<double> quat(4, 0.0);

            while( std::getline(fp, line) ){
                vector<string> words;
                boost::split(words, line, boost::is_any_of(" "));
                if(words[0].compare("VERTEX_SE3:QUAT") == 0){
                    vertices++;
                    int node_index = boost::lexical_cast<int>(words[1]);
                    for(int i = 0; i<3; i++){
                        pos[i] = boost::lexical_cast<double>(words[i+2]);
                        cout<<pos[i]<<" ";     
                    }
                    
                    for(int i = 0; i<4; i++){
                        quat[i] = boost::lexical_cast<double>(words[i+5]);
                        cout<<quat[i]<<" ";
                    }
                    cout<<endl;
                    
                    Pose3d *node = new Pose3d(node_index, pos, quat);
                    nodes.push_back(node);
                }
                else if(words[0].compare("EDGE_SE3:QUAT") == 0){
                    int idx_a = boost::lexical_cast<int>(words[1]);
                    int idx_b = boost::lexical_cast<int>(words[2]);
                    
                    vector<double> pos(3, 0.0);
                    vector<double> quat(4, 0.0);
                    for(int i = 0; i<3; i++){
                        pos[i] = boost::lexical_cast<double>(words[i+3]);     
                    }
                    for(int i = 0; i<4; i++){
                        quat[i] = boost::lexical_cast<double>(words[i+6]);
                    }

                    if(abs(idx_a - idx_b) > 0){
                        Edge *edge = new Edge(nodes[idx_a], nodes[idx_b], ODOMETRY_EDGE);
                        edge->setEdgeTransform(pos, quat);
                        odometry_edges.push_back(edge);
                    }
                    else{
                        Edge *edge = new Edge(nodes[idx_a], nodes[idx_b], CLOSURE_EDGE);
                        edge->setEdgeTransform(pos, quat);
                        closure_edges.push_back(edge);
                    }
                }
            }
        }

        void writePoseGraph_nodes(const string& fname){
            cout << "Write Pose Graph Nodes: " << fname << endl;
            fstream fp;
            fp.open(fname.c_str(), ios::out);
            for( int i=0; i<this->nodes.size(); i++){
                fp << nodes[i]->index << " "<<
                nodes[i]->pos.x() <<" "<< nodes[i]->pos.y() <<" "<<nodes[i]->pos.z() << endl;
            }
        }

        void writePoseGraph_edges(const string& fname){
            cout<< "Writing Pose Graph Edges: "<< fname << endl;
            fstream fp;
            fp.open(fname.c_str(), ios::out);
            write_edges(fp, this->odometry_edges);
            write_edges(fp, this->closure_edges);
        }

        void write_edges(fstream &fp, vector<Edge*> &vec){
            for(int i=0; i<vec.size(); i++){
                fp<<vec[i]->a->index<<" "<<vec[i]->b->index<<" "<<vec[i]->edge_type<<endl;
            }
        }
};


struct PoseResidue{
    Eigen::Matrix<double, 3, 1> dpos;
    Eigen::Quaternion<double> dq;
    PoseResidue(const Edge *edge){
        this->dpos = edge->dpos;
        this->dq  = edge->dquat;
    }

    template<typename T>
    bool operator()(const T* const pos1, const T* const quat1, 
                    const T* const pos2, const T* const quat2, T* e) const{

        //P1 to T1^ w_T_a
        Eigen::Map<const Eigen::Quaternion<T>> q_a(quat1);
        Eigen::Map<const Eigen::Matrix<T,3,1>> p_a(pos1);

        //P2 to T2^ w_T_b
        Eigen::Map<const Eigen::Quaternion<T>> q_b(quat2);
        Eigen::Map<const Eigen::Matrix<T,3,1>> p_b(pos2);

        // now we have :: w_T_a, w_T_b and a_Tcap_b
        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
        Eigen::Quaternion<T> q_ab_estimated = (q_a_inverse * q_b);
        Eigen::Quaternion<T> diff_q = dq.template cast<T>() * q_ab_estimated.conjugate(); 
        Eigen::Matrix<T, 3, 1> diff_pos = dpos.template cast<T>()  - (q_a_inverse * (p_b - p_a));

        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(e);
        residuals.template block<3,1>(0,0) = diff_pos;
        residuals.template block<3,1>(3,0) = T(2.0) * diff_q.vec();
       
        return true;
    }

    static ceres::CostFunction* Create(const Edge *edge){
         return (new ceres::AutoDiffCostFunction<PoseResidue, 6, 3, 4, 3, 4>(
            new PoseResidue(edge)));
    };
};


//Assumes .g20 file is in the same directory
int main(){

    //string fname = "../input_M3500_g2o.g2o";
    //01string fname = "../parking-garage.g2o";
    string fname = "../sphere_bignoise_vertex3.g2o";
    cout<< "Reading Pose Graph "<< fname <<endl;
    ReadG20 g(fname);

    string output_fname = "../before_opt.txt";
    //g.writePoseGraph_edges(output_fname);
    g.writePoseGraph_nodes(output_fname);

    cout<< "Total Nodes : " << g.nodes.size() << endl;
    cout<< "Total Odometry edges : " << g.odometry_edges.size() << endl;
    cout<< "Total Closure edges : " << g.closure_edges.size() << endl;
    
    ceres::Problem problem;
    ceres::LossFunction* loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization =
      new EigenQuaternionParameterization;
    for(int i=0; i<g.odometry_edges.size(); i++){
        Edge *ed = g.odometry_edges[i];
        ceres::CostFunction *cost_function = PoseResidue::Create(ed);
        
        problem.AddResidualBlock(cost_function, loss_function, ed->a->pos.data(), ed->a->quat.coeffs().data(), 
                                ed->b->pos.data(), ed->b->quat.coeffs().data());
        
        problem.SetParameterization(ed->a->quat.coeffs().data(),
                                 quaternion_local_parameterization);
        problem.SetParameterization(ed->b->quat.coeffs().data(),
                                 quaternion_local_parameterization);
        double res[6];
        double *params[4];

        params[0] = ed->a->pos.data();
        params[1] = ed->a->quat.coeffs().data();
        params[2] = ed->b->pos.data();
        params[3] = ed->b->quat.coeffs().data();

        cost_function->Evaluate(params, res, NULL);
        cout<<"Edge Cost: "<<res[0]<<" "<<res[1]<<" "<<res[2]<<" "<<res[3]<<" "<<res[4]<<" "<<res[5]<<endl;
    }

    //Setting first node as the prior
    problem.SetParameterBlockConstant(g.nodes[0]->pos.data());
    problem.SetParameterBlockConstant(g.nodes[0]->quat.coeffs().data());

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;

    // Write Pose Graph after Optimization
    g.writePoseGraph_nodes("../after_opt.txt");
    
    return 0;
}
 