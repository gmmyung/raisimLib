//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "../../RaisimGymEnv.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <pybind11/pybind11.h>
#include <set>
#include <stdlib.h>

namespace py = pybind11;

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

public:
  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg,
                       bool visualizable)
      : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable),
        normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ =
        world_->addArticulatedSystem(resourceDir_ + "/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4,
        -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 47;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    obs_body_pos_.setZero(3);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std,
              cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile(cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final {}

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec> &action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_)
        server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_)
        server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2],                 /// body height
        gc_.segment(3, 4),               /// body orientation quaternion
        gc_.tail(12),                    /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12),                    /// joint velocity
        anymal_->getGeneralizedForce().e().tail(12); /// joint force
    ;
  }

  void setObservation(const Eigen::Ref<EigenVec> &ob, bool init) {
    // define zero matrix obs_gc
    obs_gc_ = Eigen::VectorXd::Zero(gcDim_);
    auto ob_double = ob.cast<double>();
    // copy body height
    obs_gc_[2] = ob_double[0];
    // copy body orientation
    obs_gc_.segment(3, 4) = ob_double.segment(1, 4);
    // copy joint angles
    obs_gc_.tail(12) = ob_double.segment(5, 12);
    // increment body x, y coordinates
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = obs_gc_[3];
    quat[1] = obs_gc_[4];
    quat[2] = obs_gc_[5];
    quat[3] = obs_gc_[6];
    raisim::quatToRotMat(quat / quat.norm(), rot);
    Eigen::Vector3d bodyLinearVel = rot.e() * ob_double.segment(17, 3);
    if (init) {
      obs_body_pos_ = Eigen::Vector3d(0, 0, 0);
    } else {
      obs_body_pos_[0] += bodyLinearVel[0] * control_dt_;
      obs_body_pos_[1] += bodyLinearVel[1] * control_dt_;
    }
    obs_gc_[0] = obs_body_pos_[0];
    obs_gc_[1] = obs_body_pos_[1];
    // set gv to zero
    obs_gv_ = Eigen::VectorXd::Zero(gvDim_);

    anymal_->setState(obs_gc_, obs_gv_);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float &terminalReward) final {
    // terminalReward = float(terminalRewardCoeff_);
    //
    // /// if the contact body is not feet
    // for (auto &contact : anymal_->getContacts())
    //   if (footIndices_.find(contact.getlocalBodyIndex()) ==
    //   footIndices_.end())
    //     return true;
    //
    // terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() {};

private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem *anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd obs_gc_, obs_gv_, obs_body_pos_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create
  /// a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

} // namespace raisim
