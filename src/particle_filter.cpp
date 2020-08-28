/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * ::: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * ::: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  if (is_initialized) {
    return;
  }
  
  // random engine generator
  std::default_random_engine gen;
  
  // ::: Set the number of particles
  num_particles = 100;  
  
  // Normal (Gaussian) distribution for x, y and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    
    // Create new particle
    Particle new_particle;
    
    double sample_x, sample_y, sample_theta;
    
    // Generate paticle values in random
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);   
    
    // Fill particle variables
    new_particle.id = i;
    new_particle.x = sample_x;
    new_particle.y = sample_y;
    new_particle.theta = sample_theta; 
    new_particle.weight = 1.0;
    
    // Fill particle and weights vector
    particles.push_back(new_particle);
    weights.push_back(1.0);
  }
  
  // Filter initialized
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * ::: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // random engine generator
  std::default_random_engine gen;
  
  // TO avoid division by zero
  if (fabs(yaw_rate) < 0.0001) {
      yaw_rate = 0.0001;
  }
  
  for (int i = 0; i < num_particles; ++i) {
    double pred_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
    double pred_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
    double pred_theta = particles[i].theta + (yaw_rate * delta_t);
    
    // Normal (Gaussian) distribution for x, y and theta
    std::normal_distribution<double> dist_x(pred_x, std_pos[0]);
    std::normal_distribution<double> dist_y(pred_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
    
    // Update particle values
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * ::: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); i++) {
    
    // initial distance to nearest prediction and its id
    int nearest_prediction_id = -1;
    double nearest_prediction_dist = 100000000;
    
    for (int j = 0; j < predicted.size(); j++) {
      double current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      if (current_dist < nearest_prediction_dist) {
        nearest_prediction_dist = current_dist;
        nearest_prediction_id = predicted[j].id;
      }
    }
    
    observations[i].id = nearest_prediction_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * ::: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (int i = 0; i < num_particles; ++i) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    
    // 1. Landmarks within the range of sensors
    vector<LandmarkObs> landmarks_in_range;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      if ( (p_x - map_landmarks.landmark_list[j].x_f < sensor_range) && (p_y - map_landmarks.landmark_list[j].y_f < sensor_range) ) {
        LandmarkObs temp;
        temp.id = map_landmarks.landmark_list[j].id_i;
        temp.x = map_landmarks.landmark_list[j].x_f;
        temp.y = map_landmarks.landmark_list[j].y_f;
        landmarks_in_range.push_back(temp);
      }
    }
    
    // 2. Transform observations from car co-ordinates to map co-ordinates
    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs map_obs;
      map_obs.id = j;
      map_obs.x = p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
      map_obs.y = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
      transformed_observations.push_back(map_obs);
    }
    
    // 3. Associate the in range land marks with transformed observations
    dataAssociation(landmarks_in_range, transformed_observations);
    
    // 4. Find weights for each particle
    double sig_x= std_landmark[0];
    double sig_y= std_landmark[1];
    particles[i].weight = 1.0;
    for (int j = 0; j < transformed_observations.size(); j++) {
      double x_obs = transformed_observations[j].x;
      double y_obs = transformed_observations[j].y;
      double mu_x;
      double mu_y;
      for (int k = 0; k < landmarks_in_range.size(); k++) {
        if(landmarks_in_range[k].id == transformed_observations[j].id) {
          mu_x = landmarks_in_range[k].x;
          mu_y = landmarks_in_range[k].y;
          break;
        }
      }
      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
      double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
      double temp_weight = gauss_norm * exp(-exponent);
      particles[i].weight = particles[i].weight * temp_weight;
    } 
  }
  
  // 5. Normalize weights
  double total_weight = 0.0;
  for (int i = 0; i < num_particles; ++i) {
    total_weight += particles[i].weight;
  }
  for (int i = 0; i < num_particles; ++i) {
    particles[i].weight = particles[i].weight / total_weight;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * ::: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // random engine generator
  std::default_random_engine gen;
  
  vector<Particle> resampled_particles;
  
  //Generate random index
  std::uniform_int_distribution<int> rand_index(0, num_particles - 1);
  int rand_gen_index = rand_index(gen);
  
  double beta = 0.0;
  double max_weight = *max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> rand_weight(0.0, max_weight);
  for(int i = 0; i < num_particles; i++) {
    beta = beta + rand_weight(gen) * 2.0;
    while(beta > weights[rand_gen_index]) {
      beta = beta - weights[rand_gen_index];
      rand_gen_index = (rand_gen_index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[rand_gen_index]);
  }
  particles = resampled_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}