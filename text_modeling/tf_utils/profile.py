# encoding=utf-8
from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

def tf_profile(sess, variable, data, times):
    for fd in data:

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)

def save_json(profiler, step, path):
    profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
    option_builder.ProfileOptionBuilder.time_and_memory())
    #输出方式为timeline
    profile_graph_opts_builder.with_timeline_output(timeline_file=path)
    #定义显示sess.Run() 第70步的统计数据
    profile_graph_opts_builder.with_step(step)
    #显示视图为graph view
    profiler.profile_graph(profile_graph_opts_builder.build())

def rank_ops_by_time(profiler):
    profile_op_opt_builder = option_builder.ProfileOptionBuilder()
    #显示字段：op执行时间，使用该op的node的数量。 注意：op的执行时间即所有使用该op的node的执行时间总和。
    profile_op_opt_builder.select(['micros','occurrence'])
    #根据op执行时间进行显示结果排序
    profile_op_opt_builder.order_by('micros')
    #过滤条件：只显示排名top 5
    profile_op_opt_builder.with_max_depth(4)
    #显示视图为op view
    profiler.profile_operations(profile_op_opt_builder.build())

def rank_ops_by_memory(profiler):
    profile_op_opt_builder = option_builder.ProfileOptionBuilder()

    #显示字段：op占用内存，使用该op的node的数量。 注意：op的占用内存即所有使用该op的node的占用内存总和。
    profile_op_opt_builder.select(['bytes','occurrence'])
    #根据op占用内存进行显示结果排序
    profile_op_opt_builder.order_by('bytes')
    #过滤条件：只显示排名最靠前的5个op
    profile_op_opt_builder.with_max_depth(4)

    #显示视图为op view
    profiler.profile_operations(profile_op_opt_builder.build())

def show_params(profiler):

    #统计内容为所有trainable Variable Op
    profile_scope_opt_builder = option_builder.ProfileOptionBuilder(
      option_builder.ProfileOptionBuilder.trainable_variables_parameter())

    #显示的嵌套深度为4
    profile_scope_opt_builder.with_max_depth(4)
    #显示字段是params，即参数
    profile_scope_opt_builder.select(['params'])
    #根据params数量进行显示结果排序
    profile_scope_opt_builder.order_by('params')

    #显示视图为scope view
    profiler.profile_name_scope(profile_scope_opt_builder.build())
