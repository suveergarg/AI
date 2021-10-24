#include<mutex>
#include<chrono>
#include<thread>
#include<iostream>

//Waiting for flag
bool flag = false;
std::mutex m;
void wait_for_flag(){
    std::unique_lock<std::mutex> lk(m);
    while(!flag){
        lk.unlock();
        std::cout<<"Waiting for Flag and sleeping for 100ms"<<std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //If threads sleeps, other threads get a chance to execute
        // It is hard to get this sleep time right for real time applications.
        lk.lock();
    }
}

void thread(int i){
    std::cout<< "Calling from Thread "<< i <<std::endl;
    ::flag = true;
}

int main(){
    std::thread t1(wait_for_flag);
    //Each thread has its own program counter, a stack, and a set of registers.
    std::thread t2(thread, 1);
    std::thread t3(thread, 2);

    t1.join();
    t2.join();
    t3.join();

    return 0;
}
