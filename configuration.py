import configparser
from configuration_section import ConfigurationSection
from configuration_option import ConfigurationOption


class Configuration:

    @staticmethod
    def update_configuration(config_section: ConfigurationSection, config_option: ConfigurationOption, new_value):
        config = configparser.ConfigParser()
        config.read("prompt_config.ini")

        section = config_section.value
        option = config_option.value

        if not config.has_section(section):
            config.add_section(section)

        config.set(section, option, new_value)

        with open('prompt_config.ini', 'w') as configfile:
            config.write(configfile)

        print("Configuration updated successfully.")

    @staticmethod
    def save_concepts(concepts_dict):
        config = configparser.ConfigParser()
        config.read("prompt_config.ini")

        concept_section = ConfigurationSection.CONCEPTS.value

        if "Concepts" in config:
            config.remove_section(concept_section)
        config.add_section(concept_section)

        # Add each concept to the Concepts section from the provided dictionary
        for concept, description in concepts_dict.items():
            config.set(concept_section, concept, description)

        # Write the updated configuration back to the file
        with open("prompt_config.ini", "w") as configfile:
            config.write(configfile)

        print("Concepts saved to prompt_config.ini successfully.")

    @staticmethod
    def update_whole_prompt(system_message, user_message, output_format):
        # Load the .ini file
        config = configparser.ConfigParser()

        config.read("prompt_config.ini")

        if not config.has_section('Prompt'):  # Replace 'SectionName' with your actual section name
            config.add_section('Prompt')  # Create the section if it doesn't exist

            # Replace the old message with the new one
        config.set('Prompt', 'system_message', system_message)
        config.set('Prompt', 'user_message', user_message)
        config.set('Prompt', 'output_format', output_format)

        # Write changes to the .ini file
        with open('prompt_config.ini', 'w') as configfile:  # Replace with your actual .ini file name
            config.write(configfile)

        print("Prompt updated successfully.")

    @staticmethod
    def read_concepts():
        config_file = 'prompt_config.ini'
        config = configparser.ConfigParser()
        config.read(config_file)

        concept_section = ConfigurationSection.CONCEPTS.value

        return {key: config.get(concept_section, key) for key in config.options(concept_section)}

    @staticmethod
    def get_prompt_part(prompt_part: ConfigurationOption):
        config = configparser.ConfigParser()

        config.read("prompt_config.ini")

        try:
            option = config.get(ConfigurationSection.PROMPT.value, prompt_part.value)
            return option
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            return "Error: Unable to retrieve prompt option."

    @staticmethod
    def get_context_retrieval_option(prompt_part: ConfigurationOption):
        config = configparser.ConfigParser()

        config.read("prompt_config.ini")

        try:
            option = config.get(ConfigurationSection.OTHER.value, prompt_part.value)
            return option
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            return "Error: Unable to retrieve context retrieval option."

    @staticmethod
    def get_concept_description(concept):
        config_file = 'prompt_config.ini'
        config = configparser.ConfigParser()
        config.read(config_file)

        section = 'Concepts'
        if config.has_section(section) and config.has_option(section, concept):
            return config.get(section, concept)
        else:
            return None
