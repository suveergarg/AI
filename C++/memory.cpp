#include<iostream>
using namespace std;
template<typename T>
T* try_allocation(unsigned long CHUNK_SIZE){
    //Trying to allocate too much memory on the heap and dealing with that;
    try{
        T *alloc = new T[CHUNK_SIZE];
        cout<<"Memory allocated Successfully"<<endl;
    }
    catch(std::bad_alloc &e){
        cout<<"Memory Allocation Failed"<<endl;
        cout<<e.what()<<endl;
    }
}

template<typename T>
T* no_throw_allocation(unsigned long CHUNK_SIZE){
    T* alloc = new(std::nothrow) T[CHUNK_SIZE];
    if (alloc){
        cout<<"Successful Allocation"<<endl;
    }
    else{
        cout<<"Allocation Failed"<<endl;
    }
    return alloc;
}

int main(){
    double* ptr1 = try_allocation<double>(0x7ffffffffffffff);
    double* ptr = no_throw_allocation<double>(0x7ffff);
    delete[] ptr1;
    delete[] ptr;
}
