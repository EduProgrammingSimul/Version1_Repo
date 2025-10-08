{{ report_title | default("PWR Control Strategy Analysis Report") }}
Generated: {{ generation_time }}

{% if controller_details %}

Validation Focus: {{ controller_details.controller_type }}
Identifier: {{ controller_details.optimized_config_or_model_identifier }}
{% endif %}

Executive Summary
(Placeholder) This section should provide a high-level overview of the analysis, key findings across all scenarios, and the main recommendations regarding controller performance, robustness, and suitability for different operational conditions (e.g., load following, disturbance rejection). Summarize which controller(s) performed best overall according to the defined criteria.

Analysis Details by Scenario
{% for scenario_name, data in scenarios_data.items() | sort %}

Scenario: {{ scenario_name }}
Description: {{ data.config.description | default('N/A') }}

Controllers Evaluated: {{ data.controller_names | join(', ') if data.controller_names else 'None' }}

{% if data.context_info %}
Context: {{ data.context_info }}
{% endif %}

Key Metrics Comparison
{% if data.metrics_table and data.metrics_table | length > 1 %}
| {% for header_cell in data.metrics_table[0] %}{{ header_cell }} | {% endfor %}
| {% for _ in data.metrics_table[0] %}--- | {% endfor %}
{% for row in data.metrics_table[1:] %}
| {% for cell in row %}{{ cell | safe }} | {% endfor %}
{% endfor %}
{% else %}

No detailed metrics data available for table display in this scenario.
{% endif %}

Note: Check column headers for units. NaN (Sim Failed) indicates the controller did not complete this scenario successfully. Inf may indicate the system did not settle.

Winner Analysis (Comparative)
{% if data.analysis.winners %}
{% for metric, winner_text in data.analysis.winners.items() | sort %}

Best {{ metric.replace('_', ' ').title() }}: {{ winner_text | safe }}
{% endfor %}
{% else %}

No comparative winner analysis available for key metrics.
{% endif %}

Recommendation: {{ data.analysis.recommendation | default("N/A") }}

Visualizations
{% if (data.plots.time_series or data.plots.metric_comparison) and (data.plots.time_series | length > 0 or data.plots.metric_comparison | length > 0) %}
| Plot Description             | Visualization                                       |
|------------------------------|-----------------------------------------------------|
{% for plot_info in data.plots.time_series %}
| {{ plot_info.title }}        | ![{{ plot_info.title }}]({{ plot_info.path }})      |
{% endfor %}
{% for plot_info in data.plots.metric_comparison %}
| {{ plot_info.title }}        | ![{{ plot_info.title }}]({{ plot_info.path }})      |
{% endfor %}
{% else %}

No automatically linked plots available for this scenario. Check the {{ report_config.plot_output_dir if report_config else 'results/plots' }} directory.
{% endif %}

{% else %}
No scenario results available to report.
{% endfor %}

Overall Summary & Conclusions
(Placeholder) This section could include:

A summary table comparing key metrics averaged across all relevant scenarios for each controller.

Discussion on controller robustness, highlighting controllers that successfully completed all scenarios versus those that failed.

Final conclusions regarding the suitability of each controller for the simulated PWR system.

Recommendations for future work or specific controller deployment strategies.

End of Report