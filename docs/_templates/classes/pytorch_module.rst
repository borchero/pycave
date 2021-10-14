:orphan:

.. role:: hidden

{{ name | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
    :show-inheritance:

{% if methods and not name.endswith("Config") %}
.. rubric:: Methods

.. autosummary::
    :toctree:
    :nosignatures:

    {% for item in methods %}
    {%- if not item in inherited_members %}
    {%- if not item.startswith("_") %}
    ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endif %}
    {%- endfor %}
{%- endif %}

{% if methods and not name.endswith("Config") %}
.. rubric:: Inherited Methods

.. autosummary::
    :toctree:
    :nosignatures:

    {% for item in methods %}
    {%- if item in ["load", "save"] %}
    ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endfor %}
{%- endif %}

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
    {% for item in attributes %}
    {%- if not item in ["training", "T_destination", "dump_patches"] %}
    ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endfor %}
{%- endif %}
