from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Voter',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('voter_id', models.CharField(max_length=50, unique=True)),
                ('face_embedding', models.TextField()),
                ('iris_features', models.TextField()),
                ('has_voted', models.BooleanField(default=False)),
            ],
        ),
    ]
