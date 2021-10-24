#include<iostream>
#include<memory>
void compute(std::unique_ptr<int[]> p){


    return;
}

int main(){

    std::unique_ptr<int[]> ptr = std::make_unique<int[]>(1024);
    //std::unique_ptr<int[]> p = ptr; // This will lead to an error
    //compute(ptr);
    //Ptr is passed by copy which is not allowed

    std::shared_ptr<int> p1(new int);
    std::shared_ptr<int> p2(p1);
}
