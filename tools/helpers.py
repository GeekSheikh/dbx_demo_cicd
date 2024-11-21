from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType
from concurrent.futures import ThreadPoolExecutor
from pyspark.ml.feature import Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

class DFHelpers:
    def __init__(self, df: DataFrame):
        self.df = df
        self.field_cardinalities = {}

    def get_field_cardinalities(self, omit_fields=None, num_threads=8):
        """
        Calculate and store the cardinalities (distinct counts) of all fields in the DataFrame,
        excluding any fields specified in the omit_fields list. This is done using multithreading.
        """
        if omit_fields is None:
            omit_fields = []

        def calculate_cardinality(field):
            if field.name not in omit_fields:
                cardinality = self.df.select(field.name).distinct().count()
                self.field_cardinalities[field.name] = cardinality

        if not self.field_cardinalities:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                executor.map(calculate_cardinality, self.df.schema.fields)

        return self.field_cardinalities

    def get_numerical_fields(self):
        """
        Identify numerical fields in the DataFrame schema.
        """
        numerical_fields = [field.name for field in self.df.schema.fields if isinstance(field.dataType, (DoubleType, IntegerType, LongType))]
        return numerical_fields

    def select_numericals(self):
        """
        Select only numerical fields from the DataFrame.
        """
        numerical_fields = self.get_numerical_fields()
        numerical_df = self.df.select(*numerical_fields)
        return numerical_df

    def get_continuous_fields(self, threshold=20, omit_fields=None):
        """
        Identify continuous fields in the DataFrame schema based on cardinality.
        Continuous fields are those that are numerical and have cardinality greater than the specified threshold.
        """
        self.get_field_cardinalities(omit_fields)
        numerical_fields = self.get_numerical_fields()
        continuous_fields = [field for field in numerical_fields if self.field_cardinalities.get(field, 0) > threshold]
        return continuous_fields

    def get_categorical_fields(self, threshold=20, omit_fields=None):
        """
        Identify categorical fields in the DataFrame schema based on cardinality.
        Categorical fields are those with cardinality less than or equal to the specified threshold.
        """
        self.get_field_cardinalities(omit_fields)
        categorical_fields = [field for field, cardinality in self.field_cardinalities.items() if cardinality <= threshold]
        return categorical_fields
    
    def _build_conditions(self, fields, op, threshold):
        """
        Build conditions for filtering based on the specified operation and threshold for the given fields.
        The operation can be one of 'gt' (greater than), 'lt' (less than), 'eq' (equal to), or 'between' (inclusive range).
        """
        conditions = []
        for field in fields:
            if op == 'gt':
                filter_condition = F.col(field) > threshold
            elif op == 'lt':
                filter_condition = F.col(field) < threshold
            elif op == 'eq':
                filter_condition = F.col(field) == threshold
            elif op == 'between':
                if not isinstance(threshold, list) or len(threshold) != 2:
                    raise ValueError("For 'between' operation, threshold should be a list of two elements [lower_bound, upper_bound].")
                filter_condition = (F.col(field) >= threshold[0]) & (F.col(field) <= threshold[1])
            else:
                raise ValueError("Invalid operation. Use 'gt', 'lt', 'eq', or 'between'.")
            conditions.append({'field': field, 'filter': filter_condition})
        return conditions

    def fields_global_filter_op(self, fields, op, threshold):
        """
        Filter the DataFrame based on the specified operation and threshold for the given fields.
        The operation can be one of 'gt' (greater than), 'lt' (less than), 'eq' (equal to), or 'between' (inclusive range).
        """
        if not isinstance(fields, list):
            raise ValueError("Fields should be a list of field names.")

        conditions = self._build_conditions(fields, op, threshold)
        combined_condition = conditions[0]['filter']
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition['filter']

        return combined_condition
    
    def apply_df_filters(self, df, field_ops, watch_filter=False):
        """
        Filter the DataFrame based on the specified operations and thresholds for the given fields.
        The field_ops should be a dictionary where keys are field names and values are dictionaries with 'op' and 'threshold'.
        The operation can be one of 'gt' (greater than), 'lt' (less than), 'eq' (equal to), or 'between' (inclusive range).
        If watch_filter is True, print the name of the field and the DataFrame count after each filter is applied.
        """
        if not isinstance(field_ops, dict):
            raise ValueError("field_ops should be a dictionary with field names as keys and dictionaries with 'op' and 'threshold' as values.")

        conditions = []
        filtered_df = df
        for field, ops in field_ops.items():
            op = ops.get('op')
            threshold = ops.get('threshold')
            conditions.extend(self._build_conditions([field], op, threshold))
            
        for condition in conditions:
            filtered_df = filtered_df.filter(condition['filter'])
            if watch_filter:
                print(f"Field: {condition['field']}, Count after filter: {filtered_df.count()}")

        return filtered_df
    
    def scale_fields(self, df, fields):
        """
        Scale the specified fields in the DataFrame using RobustScaler.
        The output field names will be the same as the input field names.
        """

        for field in fields:
            assembler = VectorAssembler(inputCols=fields, outputCol="features")
            scaler = RobustScaler(inputCol="features", outputCol="scaled_features",
                        withScaling=True, withCentering=False,
                        lower=0.001, upper=0.999)

            pipeline = Pipeline(stages=[assembler, scaler])
            scaled_df = pipeline.fit(df).transform(df)

            for i, field in enumerate(fields):
                scaled_df = scaled_df.withColumn(field, vector_to_array(col("scaled_features")).getItem(i))

            scaled_df = scaled_df.drop("features", "scaled_features")
        
        return scaled_df