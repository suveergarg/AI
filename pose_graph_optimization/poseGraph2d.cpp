#include <iostream>
#include <string>
#include <fstream>

#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;

//Reference : https://github.com/mpkuse/toy-pose-graph-optimization-ceres/blob/master/ceres_vertigo.cpp 

//Store types of edges
enum EdgeType{
    ODOMETRT_EDGE = 0,
    CLOSURE_EDGE,
};


class Node
{
    const double x, y, theta;
    const int index;

    public:

    Node(int index, double x, double y, double theta): 
        index(index), 
        x(x),
        y(y),
        theta(theta) {}
};

class Edge
{
    EdgeType edge_type;
    const Node *a, *b;
    double x,y,theta;
    public:
    Edge(const Node* a, const Node *b, EdgeType edge_type):
    a(a), b(b), edge_type(edge_type){}
};


class ReadG20
{
    vector<Node*> nodes;
    public:
        ReadG20(const string &fName){
            fstream fp;
            fp.open(fName.c_str(), ios::in);

            string line;
            int vertices;
            int edges;

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

            }
        }
};


int main(){
    return 0;
}