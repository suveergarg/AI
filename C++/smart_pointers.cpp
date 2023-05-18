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

int main(){

    std::unique_ptr<Rectangle> ptr(new Rectangle(10, 5));
    cout<<ptr->area()<<endl;
    unique_ptr<Rectangle> ptr2;
    ptr2 = move(ptr);
    cout<<ptr2->area()<<endl;

    //The following line will throw an error
    //cout<<ptr->area()<<endl;
    //func1(ptr2);

    std::shared_ptr<int> p1(new int(70));
    cout<<"p1 :"<<p1.use_count()<<endl;
    std::shared_ptr<int> p2(p1);
    cout<<"p1 :"<<p1.use_count()<<endl;
    
    func(p1);
    cout<<"p1 :"<<p1.use_count()<<endl;
    cout<<*p1<<endl;
    cout<<*p2<<endl;

    return -1;

}
