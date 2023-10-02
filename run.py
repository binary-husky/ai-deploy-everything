# import necessary packages
import void_terminal as vt
from rich.live import Live
from rich.markdown import Markdown

# ⭐ configure
# set API_KEY configuration
vt.set_conf(key="API_KEY", value="fk195831-xxxxx")
# set LLM_MODEL configuration
vt.set_conf(key="LLM_MODEL", value="gpt-3.5-turbo")
# set USE_PROXY configuration
vt.set_conf(key="USE_PROXY", value="True")
# set proxies configuration
vt.set_conf(key="proxies", value='{ "http": "http://localhost:10881", "https": "http://localhost:10881" }' )

# define a function for plugin shortcut
def plugin_shortcut(main_input, plugin, advanced_arg=None, debug=False):
    # import necessary packages
    from rich.live import Live
    from rich.markdown import Markdown
    cookies = None
    while True:
        next_round = None
        # create a Live object for displaying output
        with Live(Markdown(""), auto_refresh=False, vertical_overflow="visible") as live:
            plugin_path = plugin if cookies is None else cookies['lock_plugin']
            plugin_h = vt.get_plugin_handle(plugin)
            plugin_kwargs = vt.get_plugin_default_kwargs()
            plugin_kwargs['main_input'] = main_input
            if cookies is not None: plugin_kwargs['chatbot_with_cookie']._cookies = cookies
            if advanced_arg is not None:
                plugin_kwargs['plugin_kwargs'] = advanced_arg
            if debug:
                my_working_plugin = plugin_h(**plugin_kwargs)
            else:
                my_working_plugin = vt.silence_stdout(plugin_h)(**plugin_kwargs)
            for cookies, chat, hist, msg in my_working_plugin:
                if isinstance(chat, dict): 
                    next_round = chat['label']
                    chat = chat['value']
                md_str = vt.chat_to_markdown_str(chat)
                md = Markdown(md_str)
                live.update(md, refresh=True)

        if cookies.get('lock_plugin', None):
            print('Fetch user feedback, and then continue.')
            main_input = input('Please input >>')
        else:
            print('All jobs done')
            break

if __name__ == "__main__":
    # run plugin shortcut function with specified parameters
    plugin_shortcut('', 'github_runner->github_runner', advanced_arg=None, debug=True)