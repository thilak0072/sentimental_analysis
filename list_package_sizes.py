import pkg_resources
import os
import shutil

def get_package_size(package_name):
    try:
        package_path = pkg_resources.resource_filename(package_name, '')
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(package_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except Exception as e:
        return str(e)

def main():
    installed_packages = pkg_resources.working_set
    package_sizes = {}
    for package in installed_packages:
        package_name = package.project_name
        package_size = get_package_size(package_name)
        package_sizes[package_name] = package_size

    for package, size in package_sizes.items():
        print(f"{package}: {size}")

if __name__ == "__main__":
    main()
