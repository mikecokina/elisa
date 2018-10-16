from conf import config
config.read_and_update_config()

print(config.CONFIG_FILE)
print(config.EXAMPLE_PARAM_1)
print(config.EXAMPLE_PARAM_2)
