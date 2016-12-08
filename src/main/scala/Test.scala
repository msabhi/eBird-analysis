/*
   val parsedData = inputRDD.map(line => {
     val fields = line.split(",")
     var sb = ""
     var index = 0
     var features = Array.ofDim[Double](columnsSet.size)
     println("SIZE =>" + columnsSet.size + " " + features.length)
     var arrayIndex = 0
     fields.foreach(col => {
       if (columnsSet.contains(index)) {
         if (isAllDigits(col)) {
           features(arrayIndex) = col.toDouble
         }
         else{
           println("INDEX =>" + arrayIndex)
           features(arrayIndex) = 1
         }
         arrayIndex = arrayIndex + 1
       }
       index = index + 1
     })
     LabeledPoint(features(0), Vectors.dense(features.tail))
   })
   */