#include "Common.hpp"
#include "cartpole.hpp"

int main() {
  raisim::World::setActivationKey("~/.raisim/activation.key");
  cartpole();
}