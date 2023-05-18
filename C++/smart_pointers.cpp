#include<iostream>
#include<memory>

using namespace std;

class Rectangle{
    int height;
    int length;

    public:

    Rectangle(int l, int h): length(l), height(h){
    }

    int area(){
        return length * height;
    }
};

void func(shared_ptr<int> ptr){
    cout<<"Here p1: "<<ptr.use_count()<<endl;
}

void func1(unique_ptr<Rectangle> ptr){
    cout<<"You should never come into this function"<<endl;
}

std::weak_ptr<int> compute(){
    shared_ptr<int> p1(new int(10));
    shared_ptr<int> p2(p1);
    weak_ptr<int> p3(p1);
    
    //This is wrong way to access contents of the weak pointer
    //cout<<"Contents of weak pointer: "<<*p3<<endl;

    //Right way to access contents of a weak pointer
    cout<<"Contents of a weak pointer: "<<*(p3.lock())<<endl;
    
    // Convert weak pointer to shared pointer
    shared_ptr<int> p_shared_orig = p3.lock();
    
    return p3;
}

int main(){

    unique_ptr<Rectangle> ptr(new Rectangle(10, 5));
    cout<<ptr->area()<<endl;
    unique_ptr<Rectangle> ptr2;
    ptr2 = move(ptr);
    cout<<ptr2->area()<<endl;

    //The following line will throw an error
    //cout<<ptr->area()<<endl;
    //func1(ptr2);

    //Shared pointer reference count gets incremented
    shared_ptr<int> p1(new int(70));
    cout<<"p1 :"<<p1.use_count()<<endl;
    shared_ptr<int> p2(p1);
    cout<<"p1 :"<<p1.use_count()<<endl;
    
    func(p1);
    cout<<"p1 :"<<p1.use_count()<<endl;
    cout<<*p1<<endl;
    cout<<*p2<<endl;

    // weak pointers
    // Does not increment reference count. 
    // Resolves cyclic dependencies (Check link for example).

    weak_ptr<int>p3(p1);
    cout<<"weak pointer p1: "<<p3.use_count()<<endl;

    weak_ptr<int> po = compute();
    cout<<"weak pointer from compute po: "<< po.use_count()<<endl;
    cout<<"po is expired: "<< po.expired() <<endl;

    return -1;

}
