#include<iostream>
#include<vector>

using namespace std;

template<typename T>
class Matrix{

    vector<vector<T>> m;
    int r;
    int c;

    public:
    Matrix(int r, int c):r(r), c(c){
        m.resize(r, vector<T>(c, 0));
    }

    Matrix(vector<vector<T>> &mat){
        this->m = mat;
        this->r = mat.size();
        this->c = mat[0].size();
    }

    friend ostream &operator<<(ostream &cout, const Matrix<T> &mat){
        for(int i=0; i<mat.r; i++){
            for(int j=0; j<mat.c; j++){
                cout<< mat.m[i][j] << " ";
            }
            cout<<endl;
        }
        return cout;
    }

    Matrix<T> operator+(const Matrix<T> mat){
        //NOTE how variables of mat do not have to be made public here
        //Matrix<T> *result = new Matrix<T>(this->r, this->c);
        Matrix<T> result(this->r, this->c);
        for(int i= 0; i<this->r; i++){
            for(int j=0; j<this->c; j++){
                result.m[i][j] = this->m[i][j] + mat.m[i][j];
            }
        }
        return result;
    }
};

int main(){

    int r = 3;
    int c = 3;
    Matrix<double> m(r,c);

    vector<vector<int>> mat(r, vector<int>(c, 7));
    vector<vector<int>> mat1(r, vector<int>(c, 2));
    Matrix<int> m1(mat);
    Matrix<int> m2(mat1);
    cout<<m1;
    cout<<m2;

    Matrix<int> m3 = m1+m2;
    cout<<m3;
    return 0;
}

