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
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "string"
            }
          ]
        },
        "rotation_period": {
          "anyOf": [
            {
              "type": "number",
              "minimum": 0.0
            },
            {
              "type": "string"
            }
          ]
        },
        "gamma": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "string"
            }
          ]
        },
        "reference_time": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "string"
            }
          ]
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
        "polar_log_g"
      ],
      "properties": {
        "mass": {"anyOf": [
          {"type": "number", "minimum": 0.0, "maximum": 150.0},
          {"type": "string"}
        ]},
        "t_eff": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
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
        "atmosphere": {
          "type": "string"
        },
        "polar_log_g": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
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
              "longitude": {"anyOf": [
                {"type": "number"},
                {"type": "string"}
              ]},
              "latitude": {"anyOf": [
                {"type": "number", "minimum": 0.0, "maximum": 180.0},
                {"type": "string"}
              ]},
              "angular_radius": {"anyOf": [
                {"type": "number"},
                {"type": "string"}
              ]},
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
            "required": [
              "l",
              "m",
              "amplitude",
              "frequency"
            ],
            "properties": {
              "l": {
                "type": "number"
              },
              "m": {
                "type": "number"
              },
              "amplitude": {
                "anyOf": [
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  }
                ]
              },
              "frequency": {
                "anyOf": [
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  }
                ]
              },
              "start_phase": {
                "anyOf": [
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  }
                ]
              },
              "mode_axis_phi": {
                "anyOf": [
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  }
                ]
              },
              "mode_axis_theta": {
                "anyOf": [
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  }
                ]
              },
              "temperature_perturbation_phase_shift": {
                "anyOf": [
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  }
                ]
              },
              "horizontal_to_radial_amplitude_ratio": {
                "type": "number",
                "minimum": 0
              },
              "temperature_amplitude_factor": {
                "type": "number"
              },
              "tidally_locked": {
                "type": "boolean"
              }
            }
          }
        }
      }
    }
  }
}
