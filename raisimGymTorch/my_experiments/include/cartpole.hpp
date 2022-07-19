#include "Common.hpp"

void cartpole() {

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  world.addGround();
  auto cartPole = world.addArticulatedSystem("/home/audrud/raisimlib/rsc/cartPole/cartpole.urdf");

  /// cartPole state
  Eigen::VectorXd jointNominalConfig(cartPole->getGeneralizedCoordinateDim());
  jointNominalConfig.setZero();
  jointNominalConfig[1] = 0.01;
  cartPole->setGeneralizedCoordinate(jointNominalConfig);
  
  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(cartPole);

  for (int i=0; i<200000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    server.integrateWorldThreadSafe();
  }

  server.killServer();
}