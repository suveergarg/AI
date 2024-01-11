#include<fstream> // std::ifstream
#include<iostream> // std::cout
#include<stdio.h> // printf
#include<cstring>

using namespace std;


void cat(istream &in, char* flag, int &counter){
        
    string line = "";

    while ( getline(in, line) ){    
        if(flag && strcmp(flag, "-n") == 0){
            counter++;
            cout<<to_string(counter)<<" "<<line<<"\n";
        }
        else{
            cout<<line<<"\n";
        }
    }
}


int main(int argc, char *argv[]){

    char *flag = nullptr;
    char *filename = nullptr;
    int counter = 0;

    for(int i = 1; i < argc; i++){
        ifstream in(argv[i], ios::binary);
        if(!in.eof() && in.fail()){
            flag = argv[i];
        }
        else{
            filename = argv[i];
            cat(in, flag, counter);
            in.close();            
        }
    }
 
    if(!filename){
        cat(cin, flag, counter);
    }

    return 0;
}