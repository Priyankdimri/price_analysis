import autogen 
# import API_KEY
import os 
os.environ(["API_KEY"])="AIzaSyDK3sWxK_YYnV3iGC6bNaSe0XP9SJ97s4k"

def main():
    config_list=autogen.config_list_from_json(
        env_or_file="API_KEY"
    )
    assistant=autogen.assistant(
        name="Assistant",
        llm_config={
            "config_list":config_list        }
        )
    user_proxy=autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_excution_config={
            "work_dir":"coding",
            
            "user_docker":False
        }
    )
    user_proxy.initiate_chat(assistant,message="plot graph of tesla and meta stock price")

if __name__=="__main__":
    main()
    
        

