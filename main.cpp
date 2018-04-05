#include <iostream>
extern int mnist_main();

int main() {
    std::cout << "start!" << std::endl;

    mnist_main();

    std::cout << "finish!" << std::endl;
    return 0;
}