{
  "type": "object",
  "required": [
    "system",
    "primary",
    "secondary"
  ],
  "properties": {
    "system": {
      "type": "object",
      "required": [
        "inclination",
        "period",
        "argument_of_periastron",
        "gamma",
        "eccentricity"
      ],
      "properties": {
        "inclination": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "period": {"anyOf": [
          {"type": "number", "minimum": 0.0},
          {"type": "string"}
        ]},
        "argument_of_periastron": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "gamma": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "eccentricity": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "primary_minimum_time": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "phase_shift": {
          "type": "number"
        },
        "additional_light": {
          "type": "number"
        }
      }
    },
    "primary": {
      "type": "object",
      "required": [
        "mass",
        "surface_potential",
        "synchronicity",
        "t_eff"
      ],
      "properties": {
        "mass": {"anyOf": [
          {"type": "number", "minimum": 0.0, "maximum": 150.0},
          {"type": "string"}
        ]},
        "surface_potential": {
          "type": "number",
          "minimum": 0.0
        },
        "synchronicity": {
          "type": "number",
          "minimum": 0.0
        },
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
  },
  "secondary": {
    "type": "object",
    "required": [
      "mass",
      "surface_potential",
      "synchronicity",
      "t_eff"
    ],
    "properties": {
      "mass": {"anyOf": [
        {"type": "number", "minimum": 0.0, "maximum": 150.0},
        {"type": "string"}
      ]},
      "surface_potential": {
        "type": "number",
        "minimum": 0.0
      },
      "synchronicity": {
        "type": "number",
        "minimum": 0.0
      },
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
