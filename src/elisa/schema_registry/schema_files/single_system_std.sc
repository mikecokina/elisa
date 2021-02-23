{
  "type": "object",
  "required": [
    "system",
    "star"
  ],
  "properties": {
    "system": {
      "type": "object",
      "required": [
        "inclination",
        "rotation_period",
        "gamma"
      ],
      "properties": {
        "inclination": {
          "type": "number"
        },
        "rotation_period": {
          "type": "number"
        },
        "gamma": {
          "type": "number"
        },
        "reference_time": {
          "type": "number"
        },
        "phase_shift": {
          "type": "number"
        },
        "additional_light": {
          "type": "number"
        }
      }
    },
    "star": {
      "type": "object",
      "required": [
        "mass",
        "t_eff",
        "gravity_darkening",
        "polar_log_g",
        "metallicity"
      ],
      "properties": {
        "mass": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 150.0
        },
        "t_eff": {
          "type": "number"
        },
        "gravity_darkening": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "discretization_factor": {
          "type": "number",
          "minimum": 0.0
        },
        "albedo": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "metallicity": {
          "type": "number"
        },
        "polar_log_g": {
          "type": "number"
        },
        "spots": {
          "type": "array",
          "items": {
            "type": "object",
            "required": [
              "longitude",
              "latitude",
              "angular_radius",
              "temperature_factor"
            ],
            "properties": {
              "longitude": {
                "type": "number"
              },
              "latitude": {
                "type": "number",
                "minimum": 0,
                "maximum": 180
              },
              "angular_radius": {
                "type": "number"
              },
              "temperature_factor": {
                "type": "number",
                "minimum": 0
              },
              "discretization_factor": {
                "type": "number",
                "minimum": 0
              }
            }
          }
        },
        "pulsations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "l": {
                "type": "number"
              },
              "m": {
                "type": "number"
              },
              "amplitude": {
                "type": "number"
              },
              "frequency": {
                "type": "number"
              },
              "start_phase": {
                "type": "number"
              },
              "mode_axis_phi": {
                "type": "number"
              },
              "mode_axis_theta": {
                "type": "number"
              }
            }
          }
        }
      }
    }
  }
}
