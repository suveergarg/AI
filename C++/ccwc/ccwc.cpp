#include<fstream> // std::ifstream
#include<iostream> // std::cout
#include<stdio.h> // printf
#include<cstring>

using namespace std;

struct Counter{
    long unsigned int bytes;
    long unsigned int characters;
    long unsigned int words;
    long unsigned int lines;

    Counter():bytes(0), characters(0), words(0), lines(0){
    }
};

Counter wc(istream &in){
    Counter counter;

    bool wordopen = false;
    char c;
    
    while (in.get(c)){    
        counter.bytes++;

        counter.characters++;
        
        if(c == '\n') counter.lines++;

        if(wordopen && isspace(c)){
            wordopen = false;
        }
        else if(!wordopen && !isspace(c)){
            wordopen = true;
            counter.words++;
        }   
    }

    return counter;
}

void printCounter(Counter counter, char* filename, char* flag){
    if(flag == nullptr){
        printf("%ld %ld %ld %s \n", counter.lines, counter.words, counter.characters, filename);
    }
    else if(strcmp(flag, "-c") == 0){
        printf("%ld \n", counter.bytes);
    }
    else if(strcmp(flag, "-l") == 0){
        printf("%ld \n", counter.lines);
    }        
    else if(strcmp(flag, "-w") == 0){
        printf("%ld \n", counter.words);
    }
    else if(strcmp(flag, "-m") == 0){
        printf("%ld \n", counter.characters);
    }
}

int main(int argc, char *argv[]){

    if(argc <= 1){
        printf("Enter a valid file name");
    }

    char *flag = nullptr;
    char *filename = nullptr;
    Counter counter;

    if(argc == 2){
        ifstream in(argv[1], ios::binary);
        if(!in.eof() && in.fail()){
            flag = argv[1];
            counter = wc(cin);
        }
        else{
            filename= argv[1];
            counter = wc(in);
            in.close();            
        }
    }
    else{
        ifstream in(argv[2], ios::binary);
        if(!in.eof() && in.fail()){
            cout<< "Error reading"<<argv[1]<<endl;
            return 0;
        }
        filename = argv[2];
        flag = argv[1];
        counter = wc(in);
        in.close();
    }

    printCounter(counter, filename, flag);
    return 0;
}