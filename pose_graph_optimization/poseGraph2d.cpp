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

class Node
{
    public:

    double x, y, theta;
    int index;
    double *p;

    Node(int index, double x, double y, double theta): 
        index(index), 
        x(x),
        y(y),
        theta(theta) {
            p = new double[3];
            p[0] = x;
            p[1] = y;
            p[2] = theta;
        }
};

class Edge
{    
    public:

    EdgeType edge_type;
    const Node *a, *b;
    double x,y,theta;
    double I11, I12, I13, I22, I23, I33;

    Edge(const Node* a, const Node *b, EdgeType edge_type):
    a(a), b(b), edge_type(edge_type){}

    void setEdgePose(double x, double y, double theta){
        this->x = x;
        this->y = y;
        this->theta = theta;
    }

    void setInformationMatrix( double I11, double  I12, double  I13, double I22, double I23, double I33 )
    {
        this->I11 = I11;
        this->I12 = I12;
        this->I13 = I13;
        this->I22 = I22;
        this->I23 = I23;
        this->I33 = I33;
    }
};

// Read G20 format, Write nodes to a file.
class ReadG20
{
    public:
        vector<Node*> nodes;
        vector<Edge*> odometry_edges;
        vector<Edge*> closure_edges;

        ReadG20(const string &fName){
            fstream fp;
            fp.open(fName.c_str(), ios::in);

            string line;
            int vertices;
            int edges;
            while( std::getline(fp, line) ){
                vector<string> words;
                boost::split(words, line, boost::is_any_of(" "));
                if(words[0].compare("VERTEX_SE2") == 0){
                    vertices++;
                    int node_index = boost::lexical_cast<int>(words[1]);
                    double x = boost::lexical_cast<double>(words[2]);
                    double y = boost::lexical_cast<double>(words[3]);
                    double theta = boost::lexical_cast<double>(words[4]);

                    Node *node = new Node(node_index, x, y, theta);
                    nodes.push_back(node);
                }
                else if(words[0].compare("EDGE_SE2") == 0){
                    int idx_a = boost::lexical_cast<int>(words[1]);
                    int idx_b = boost::lexical_cast<int>(words[2]);

                    double dx = boost::lexical_cast<double>(words[3]);
                    double dy = boost::lexical_cast<double>(words[4]);
                    double dtheta = boost::lexical_cast<double>(words[5]);

                    double I11, I12, I13, I22, I23, I33;
                    I11 = boost::lexical_cast<double>( words[6] );
                    I12 = boost::lexical_cast<double>( words[7] );
                    I13 = boost::lexical_cast<double>( words[8] );
                    I22 = boost::lexical_cast<double>( words[9] );
                    I23 = boost::lexical_cast<double>( words[10] );
                    I33 = boost::lexical_cast<double>( words[11] );
                

                    if(abs(idx_a - idx_b) > 0){
                        Edge *edge = new Edge(nodes[idx_a], nodes[idx_b], ODOMETRY_EDGE);
                        edge->setInformationMatrix(I11, I12, I13, I22, I23, I33);
                        edge->setEdgePose(dx, dy, dtheta);
                        odometry_edges.push_back(edge);
                    }
                    else{ 
                        Edge *edge = new Edge(nodes[idx_a], nodes[idx_b], CLOSURE_EDGE);
                        edge->setInformationMatrix(I11, I12, I13, I22, I23, I33);
                        edge->setEdgePose(dx, dy, dtheta);
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
                nodes[i]->p[0] <<" "<< nodes[i]->p[1] <<" "<<nodes[i]->p[2] << endl;
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
    double dx, dy, dtheta;
    Eigen::Matrix<double, 3,3> a_Tcap_b;
    PoseResidue(double dx, double dy, double dtheta){
        this->dx = dx;
        this->dy = dy;
        this->dtheta = dtheta;

        double cos_t = cos(this->dtheta);
        double sin_t = sin(this->dtheta);

        a_Tcap_b(0,0) = cos_t;
        a_Tcap_b(0,1) = -sin_t;
        a_Tcap_b(1,0) = sin_t;
        a_Tcap_b(1,1) = cos_t;
        a_Tcap_b(0,2) = this->dx;
        a_Tcap_b(1,2) = this->dy;

        a_Tcap_b(2,0) = 0.0;
        a_Tcap_b(2,1) = 0.0;
        a_Tcap_b(2,2) = 1.0;
    }

    template<typename T>
    bool operator()(const T* const P1, const T* const P2, T* e) const{
        //P1 to T1^ w_T_a
        Eigen::Matrix<T, 3, 3> w_T_a;
        T cos_t = T(cos(P1[2]));
        T sin_t = T(sin(P1[2]));
        w_T_a(0,0) = cos_t;
        w_T_a(0,1) = -sin_t;
        w_T_a(1,0) = sin_t;
        w_T_a(1,1) = cos_t;
        w_T_a(0,2) = P1[0];
        w_T_a(1,2) = P1[1];

        w_T_a(2,0) = T(0.0);
        w_T_a(2,1) = T(0.0);
        w_T_a(2,2) = T(1.0);

        //P2 to T2^ w_T_b
        Eigen::Matrix<T, 3, 3> w_T_b;
        cos_t = T(cos(P2[2]));
        sin_t = T(sin(P2[2]));
        w_T_b(0,0) = cos_t;
        w_T_b(0,1) = -sin_t;
        w_T_b(1,0) = sin_t;
        w_T_b(1,1) = cos_t;
        w_T_b(0,2) = P2[0];
        w_T_b(1,2) = P2[1];

        w_T_b(2,0) = T(0.0);
        w_T_b(2,1) = T(0.0);
        w_T_b(2,2) = T(1.0);

        Eigen::Matrix<T, 3, 3> T_a_Tcap_b;
        T_a_Tcap_b <<   T(a_Tcap_b(0,0)), T(a_Tcap_b(0,1)),T(a_Tcap_b(0,2)),
                        T(a_Tcap_b(1,0)), T(a_Tcap_b(1,1)),T(a_Tcap_b(1,2)),
                        T(a_Tcap_b(2,0)), T(a_Tcap_b(2,1)),T(a_Tcap_b(2,2));
        // now we have :: w_T_a, w_T_b and a_Tcap_b
        Eigen::Matrix<T,3,3> diff = T_a_Tcap_b.inverse() * (w_T_a.inverse() * w_T_b);

        e[0] = diff(0,2);
        e[1] = diff(1,2);
        e[2] = asin( diff(1,0) );

        return true;
    }

    static ceres::CostFunction* Create(const double dx, const double dy, const double dtheta){
        return (new ceres::AutoDiffCostFunction<PoseResidue, 3, 3, 3>(
            new PoseResidue(dx, dy, dtheta)));
    };
};

//Assumes .g20 file is in the same directory
int main(){

    string fname = "../input_M3500_g2o.g2o";
    cout<< "Reading Pose Graph "<< fname <<endl;
    ReadG20 g(fname);

    string output_fname = "../init_node.txt";
    //g.writePoseGraph_edges(output_fname);
    g.writePoseGraph_nodes(output_fname);

    cout<< "Total Nodes : " << g.nodes.size() << endl;
    cout<< "Total Odometry edges : " << g.odometry_edges.size() << endl;
    cout<< "Total Closure edges : " << g.closure_edges.size() << endl;

    ceres::Problem problem;
    for(int i=0; i<g.odometry_edges.size(); i++){
        Edge *ed = g.odometry_edges[i];
        ceres::CostFunction *cost_function = PoseResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.01), ed->a->p, ed->b->p);

        double res[3];
        double *params[2];

        params[0] = ed->a->p;
        params[1] = ed->b->p;

        cost_function->Evaluate(params, res, NULL);
        cout<<"Edge Cost: "<<res[0]<<" "<<res[1]<<" "<<res[2]<<endl;
    }

    //Setting first node as the prior
    problem.SetParameterBlockConstant(g.nodes[0]->p); 

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
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