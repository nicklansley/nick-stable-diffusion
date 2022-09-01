import yaml

with open("environment.yaml") as file_handle:
    environment_data = yaml.load(file_handle, Loader=yaml.FullLoader)
    print(environment_data)

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        package_name, package_version = dependency.split("=")
        file_handle.write("{} == {}".format(package_name, package_version))